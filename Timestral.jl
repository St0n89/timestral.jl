# Timestral.jl — a Julia-native, dependency-free port of Mistral's Timestral
# time-series foundation model (inference only).
#
# Reimplemented from the official Apache-2.0 reference implementation
# (https://github.com/mistralai/timestral). Only the Julia standard library is
# used. Model weights are loaded from exported checkpoints
# (config.json + model.safetensors) by local path or Hugging Face repo id.
#
# Usage:
#   include("Timestral.jl"); using .Timestral
#   fc  = Timestral.from_pretrained("mistralai/Timestral-5M")  # or a local dir
#   f   = Timestral.forecast(fc, context; prediction_length = 96)
#   f.quantiles        # (B, V, H, Q)
#   Timestral.quantile_at(f, 0.9)
#
# License: this port reimplements published reference semantics (Apache-2.0).
# The checkpoints keep their own licenses (see the Hugging Face model cards).

module Timestral

export TimestralConfig, TimestralForecaster, TimestralModel, Forecast,
       from_pretrained, forecast, quantile_at, quantile_levels, json_parse,
       encode_categorical

using Downloads
using Mmap

#=====================================================================================
  Minimal JSON (stdlib only; null -> nothing)
=====================================================================================#
mutable struct _JsonParser
    s::String
    i::Int
end

_json_ws(p) = (while p.i <= ncodeunits(p.s)
    c = codeunit(p.s, p.i)
    (c == 0x20 || c == 0x09 || c == 0x0a || c == 0x0d) || break
    p.i += 1
end)

function _json_lit(p, word)
    p.i - 1 + ncodeunits(word) <= ncodeunits(p.s) || error("bad JSON literal")
    for k in 1:ncodeunits(word)
        codeunit(p.s, p.i + k - 1) == codeunit(word, k) || error("bad JSON literal")
    end
    p.i += ncodeunits(word)
end

function _json_string(p)
    p.s[p.i] == '"' || error("expected string in JSON")
    p.i += 1
    out = IOBuffer()
    while true
        p.i <= ncodeunits(p.s) || error("unterminated JSON string")
        c, p.i = codeunit(p.s, p.i), p.i + 1
        if c == UInt8('"'); break; end
        if c == UInt8('\\')
            p.i <= ncodeunits(p.s) || error("bad escape")
            e, p.i = codeunit(p.s, p.i), p.i + 1
            if e == UInt8('"');     write(out, '"')
            elseif e == UInt8('\\'); write(out, '\\')
            elseif e == UInt8('/');  write(out, '/')
            elseif e == UInt8('b');  write(out, '\b')
            elseif e == UInt8('f');  write(out, '\f')
            elseif e == UInt8('n');  write(out, '\n')
            elseif e == UInt8('r');  write(out, '\r')
            elseif e == UInt8('t');  write(out, '\t')
            elseif e == UInt8('u')
                hex = SubString(p.s, p.i, p.i + 3); p.i += 4
                cp = parse(Int, hex; base = 16)
                if 0xD800 <= cp <= 0xDBFF && p.i + 6 <= ncodeunits(p.s) + 1 &&
                   p.s[p.i] == '\\' && p.s[p.i+1] == 'u'
                    lo = parse(Int, SubString(p.s, p.i + 2, p.i + 5); base = 16)
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    p.i += 6
                end
                write(out, Char(cp))
            else
                error("bad JSON escape")
            end
        else
            write(out, Char(c))
        end
    end
    return String(take!(out))
end

function _json_number(p)
    start = p.i
    while p.i <= ncodeunits(p.s)
        c = codeunit(p.s, p.i)
        (c == UInt8(',') || c == UInt8(']') || c == UInt8('}') ||
         c == UInt8(' ') || c == UInt8('\t') || c == UInt8('\n') || c == UInt8('\r')) && break
        p.i += 1
    end
    txt = SubString(p.s, start, p.i - 1)
    isempty(txt) && error("bad JSON number")
    v = tryparse(Int64, txt)
    return v === nothing ? parse(Float64, txt) : v
end

function _json_value(p)
    _json_ws(p)
    p.i <= ncodeunits(p.s) || error("unexpected end of JSON")
    c = p.s[p.i]
    if c == '{'
        p.i += 1
        d = Dict{String,Any}()
        _json_ws(p)
        if p.s[p.i] == '}'; p.i += 1; return d; end
        while true
            _json_ws(p)
            k = _json_string(p)
            _json_ws(p)
            p.s[p.i] == ':' || error("expected ':' in JSON object")
            p.i += 1
            d[k] = _json_value(p)
            _json_ws(p)
            if p.s[p.i] == ','; p.i += 1; continue; end
            p.s[p.i] == '}' || error("expected ',' or '}' in JSON object")
            p.i += 1
            return d
        end
    elseif c == '['
        p.i += 1
        a = Any[]
        _json_ws(p)
        if p.s[p.i] == ']'; p.i += 1; return a; end
        while true
            push!(a, _json_value(p))
            _json_ws(p)
            if p.s[p.i] == ','; p.i += 1; continue; end
            p.s[p.i] == ']' || error("expected ',' or ']' in JSON array")
            p.i += 1
            return a
        end
    elseif c == '"'
        return _json_string(p)
    elseif c == 't'; _json_lit(p, "true");  return true
    elseif c == 'f'; _json_lit(p, "false"); return false
    elseif c == 'n'; _json_lit(p, "null");  return nothing
    else
        return _json_number(p)
    end
end

"""
    json_parse(s) -> Any

Parse a JSON string (or the contents of a file read as a string) into plain
Julia values. JSON `null` becomes `nothing`.
"""
function json_parse(s::AbstractString)
    p = _JsonParser(String(s), 1)
    v = _json_value(p)
    _json_ws(p)
    p.i <= ncodeunits(p.s) && error("trailing characters after JSON value")
    return v
end

#=====================================================================================
  safetensors loader (F32/F64)
=====================================================================================#
# Weight arrays can be mmap-backed (zero-copy, file-backed clean pages that the
# kernel may reclaim under memory pressure instead of OOM-killing the process —
# what makes large checkpoints such as 209M load in small memory). The mappings
# are kept alive through this module-level cache, so weight arrays stay valid
# for the lifetime of the module.
const _mmap_keepalive = Any[]

function load_safetensors(path::AbstractString, ::Type{T}) where {T<:AbstractFloat}
    io = open(path, "r")
    n = read(io, UInt64)
    hdr = json_parse(String(read(io, Int(n))))
    base = Int(n) + 8      # data buffer starts after the 8-byte header length + header
    out = Dict{String,Array{T}}()
    for (k, v) in hdr
        k == "__metadata__" && continue
        shape = Int.(v["shape"])
        dt = v["dtype"]
        elT = dt == "F32" ? Float32 : dt == "F64" ? Float64 :
              error("unsupported safetensors dtype $dt in $k")
        nitems = prod(shape; init = 1)
        pos = base + Int(v["data_offsets"][1])
        flat = if elT === T
            # zero-copy: mmap the tensor straight out of the file
            f = nitems == 0 ? T[] :
                Mmap.mmap(io, Vector{T}, (nitems,), pos; grow = false)
            push!(_mmap_keepalive, f)   # keep the mapping alive (reshapes alias it)
            f
        else
            seek(io, pos)
            f = Vector{elT}(undef, nitems)
            read!(io, f)
            convert(Vector{T}, f)
        end
        if isempty(shape) || length(shape) == 1
            out[k] = flat
        elseif length(shape) == 2
            # The file stores C-order (d0, d1) = torch (out, in).
            # Julia column-major (in, out) is its transpose: A[i, o] = flat[o*in + i].
            out[k] = reshape(flat, shape[2], shape[1])
        else
            error("unsupported tensor rank $(length(shape)) for $k")
        end
    end
    push!(_mmap_keepalive, io)   # keep the fd alive behind the mappings
    return out
end

#=====================================================================================
  Config
=====================================================================================#
const LAYER_KIND_AXES = Dict{String,Vector{Bool}}(
    "time"          => Bool[true],
    "variate"       => Bool[false],
    "time_variate"  => Bool[true, false],
    "variate_time"  => Bool[false, true],
)

const ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV = 0, 1, 2

struct TimestralConfig
    patch_size::Int
    d_model::Int
    num_heads::Int
    mlp_hidden_dim::Int
    block_schedule::Vector{String}
    quantile_knots::Vector{Float64}
    block_size::Int
    context_length::Int
    rope_theta::Float64
    rope_partial_factor::Float64
    norm_eps::Float64
    attn_bias::Bool
    mlp_bias::Bool
    time_attn_gate::String
    variate_attn_gate::String
    minimum_scale::Float64
    use_asinh::Bool
    stabilize_with_global::Bool
    covariate_enabled::Bool
    split_context_horizon_attn::Bool
    attend_missing_patches::Bool
    quantile_width_factor::Float64
end

function config_from_dict(d::Dict)
    known = ["patch_size", "d_model", "num_heads", "mlp_hidden_dim", "block_schedule",
             "quantile_knots", "block_size", "context_length", "rope_theta",
             "rope_partial_factor", "norm_eps", "attn_bias", "mlp_bias",
             "time_attn_gate", "variate_attn_gate", "minimum_scale", "use_asinh",
             "stabilize_with_global", "covariate_enabled", "split_context_horizon_attn",
             "attend_missing_patches", "quantile_width_factor"]
    unknown = setdiff(collect(keys(d)), known)
    isempty(unknown) || error("unknown config keys: $unknown")
    getk(k, def) = haskey(d, k) ? d[k] : def
    cfg = TimestralConfig(
        Int(d["patch_size"]), Int(d["d_model"]), Int(d["num_heads"]),
        Int(d["mlp_hidden_dim"]), String.(d["block_schedule"]),
        Float64.(d["quantile_knots"]), Int(d["block_size"]), Int(d["context_length"]),
        Float64(getk("rope_theta", 10000.0)), Float64(getk("rope_partial_factor", 0.5)),
        Float64(getk("norm_eps", 1e-6)), Bool(getk("attn_bias", false)),
        Bool(getk("mlp_bias", false)), String(getk("time_attn_gate", "sdpa_sigmoid")),
        String(getk("variate_attn_gate", "sdpa_sigmoid")),
        Float64(getk("minimum_scale", 1e-6)), Bool(getk("use_asinh", true)),
        Bool(getk("stabilize_with_global", false)), Bool(getk("covariate_enabled", true)),
        Bool(getk("split_context_horizon_attn", true)),
        Bool(getk("attend_missing_patches", true)),
        Float64(getk("quantile_width_factor", 2.0)))
    cfg.d_model % cfg.num_heads == 0 ||
        error("d_model $(cfg.d_model) not divisible by num_heads $(cfg.num_heads)")
    return cfg
end

head_dim(c::TimestralConfig) = c.d_model ÷ c.num_heads
rope_dim(c::TimestralConfig) = Int(head_dim(c) * c.rope_partial_factor)
num_quantiles(c::TimestralConfig) = length(c.quantile_knots)
median_index(c::TimestralConfig) = something(findfirst(==(0.5), c.quantile_knots), 0)

function supports_known_covariates(c::TimestralConfig)
    c.covariate_enabled || return false
    ops = Bool[]
    for kind in c.block_schedule
        append!(ops, LAYER_KIND_AXES[kind])
    end
    for i in 1:length(ops)
        rest = ops[i+1:end]
        ops[i] && any(rest) && !all(rest) && return true
    end
    return false
end

#=====================================================================================
  Layers
=====================================================================================#
struct Linear{T}
    W::Matrix{T}          # (in, out): y = x * W + b  (torch stores (out, in))
    b::Vector{T}          # bias (empty when absent)
    hasbias::Bool
end
Linear{T}(W::Matrix{T}) where T = Linear{T}(W, Vector{T}(), false)

function linear3(L::Linear{T}, x::Array{T,3}) where T
    n, s, Din = size(x)
    Dout = size(L.W, 2)
    y = Array{T}(undef, n, s, Dout)
    if L.hasbias
        for i in 1:n, j in 1:s
            xi = @views x[i, j, :]
            for o in 1:Dout
                acc = L.b[o]
                @inbounds @simd for k in 1:Din
                    acc += xi[k] * L.W[k, o]
                end
                y[i, j, o] = acc
            end
        end
    else
        for i in 1:n, j in 1:s
            xi = @views x[i, j, :]
            for o in 1:Dout
                acc = zero(T)
                @inbounds @simd for k in 1:Din
                    acc += xi[k] * L.W[k, o]
                end
                y[i, j, o] = acc
            end
        end
    end
    return y
end

struct RMSNorm{T}
    scale::Vector{T}
    eps::Float64
end

function rmsnorm(n::RMSNorm{T}, x::Array{T,3}) where T
    D = size(x, 3)
    y = similar(x)
    inv = Vector{T}(undef, size(x, 1) * size(x, 2))
    for i in 1:size(x, 1), j in 1:size(x, 2)
        acc = zero(Float64)
        @inbounds @simd for k in 1:D
            acc += Float64(x[i, j, k])^2
        end
        inv[(i-1)*size(x,2)+j] = 1 / sqrt(acc / D + n.eps)
    end
    for i in 1:size(x, 1), j in 1:size(x, 2), k in 1:D
        y[i, j, k] = x[i, j, k] * inv[(i-1)*size(x,2)+j] * n.scale[k]
    end
    return y
end

_silu(x) = x / (1 + exp(-x))
_softplus(x) = x > 33 ? x : log1p(exp(x))

struct ResidualBlock{T}
    hidden::Linear{T}
    output::Linear{T}
    residual::Linear{T}
end

function (rb::ResidualBlock)(x::Array{T,3}) where T
    h = linear3(rb.hidden, x)
    for i in eachindex(h); h[i] = _silu(h[i]); end
    out = linear3(rb.output, h)
    res = linear3(rb.residual, x)
    return out .+ res
end

struct SwiGLU{T}
    w12::Linear{T}
    w3::Linear{T}
end

function (m::SwiGLU)(x::Array{T,3}) where T
    y = linear3(m.w12, x)                 # (n, s, 2*hidden): first half gate, second up
    hid = size(y, 3) ÷ 2
    for i in 1:size(y, 1), j in 1:size(y, 2), h in 1:hid
        y[i, j, h] = y[i, j, h] * _silu(y[i, j, hid + h])
    end
    return linear3(m.w3, y[:, :, 1:hid])  # only the fused (gate) half feeds w3
end

#=====================================================================================
  Weights container
=====================================================================================#
struct Attention{T}
    wQKV::Linear{T}
    wO::Linear{T}
    gate::Union{Linear{T},Nothing}
    num_heads::Int
    head_dim::Int
    is_time::Bool
end

struct TransformerBlock{T}
    attn_norms::Vector{RMSNorm{T}}
    attns::Vector{Attention{T}}
    mlp_norm::RMSNorm{T}
    mlp::SwiGLU{T}
end

struct TimestralModel{T}
    config::TimestralConfig
    patch_proj::ResidualBlock{T}
    forecast_query::Vector{T}
    blocks::Vector{TransformerBlock{T}}
    out_norm::RMSNorm{T}
    head::ResidualBlock{T}    # QuantileHead.projection
end

function TimestralModel(cfg::TimestralConfig, w::Dict{String,<:Array{<:AbstractFloat}})
    T = eltype(values(w) |> first)
    W(k) = convert(Matrix{T}, w[k])
    B(k) = convert(Vector{T}, w[k])
    lin(k) = haskey(w, k * ".bias") ? Linear{T}(W(k * ".weight"), B(k * ".bias"), true) :
                                  Linear{T}(W(k * ".weight"))
    patch_proj = ResidualBlock{T}(lin("representation.patch_proj.hidden_layer"),
                                  lin("representation.patch_proj.output_layer"),
                                  lin("representation.patch_proj.residual_layer"))
    blocks = TransformerBlock{T}[]
    for (i, kind) in enumerate(cfg.block_schedule)
        axes = LAYER_KIND_AXES[kind]
        norms = RMSNorm{T}[]
        attns = Attention{T}[]
        for (j, is_time) in enumerate(axes)
            pre = "transformer.layers.$(i-1).attn_norms.$(j-1)"
            gate = cfg.attn_bias ? (is_time ? cfg.time_attn_gate : cfg.variate_attn_gate) : "none"
            # gates have no bias in released checkpoints (attn_bias=false); keep the check
            gk = "transformer.layers.$(i-1).attns.$(j-1)._gate.weight"
            g = haskey(w, gk) ? Linear{T}(W(gk)) : nothing
            push!(norms, RMSNorm{T}(B(pre * ".scale"), cfg.norm_eps))
            push!(attns, Attention{T}(lin("transformer.layers.$(i-1).attns.$(j-1).wQKV"),
                                       lin("transformer.layers.$(i-1).attns.$(j-1).wO"),
                                       g, cfg.num_heads, head_dim(cfg), is_time))
        end
        push!(blocks, TransformerBlock{T}(
            norms, attns,
            RMSNorm{T}(B("transformer.layers.$(i-1).mlp_norm.scale"), cfg.norm_eps),
            SwiGLU{T}(lin("transformer.layers.$(i-1).mlp.w12"),
                      lin("transformer.layers.$(i-1).mlp.w3"))))
    end
    return TimestralModel{T}(
        cfg, patch_proj, convert(Vector{T}, w["representation.forecast_query"]), blocks,
        RMSNorm{T}(B("transformer.out_norm.scale"), cfg.norm_eps),
        ResidualBlock{T}(lin("head.projection.hidden_layer"),
                         lin("head.projection.output_layer"),
                         lin("head.projection.residual_layer")))
end

#=====================================================================================
  RoPE
=====================================================================================#
function rope_inv_freq(cfg::TimestralConfig)
    hd = head_dim(cfg)
    rd = rope_dim(cfg)
    inv = zeros(Float64, hd ÷ 2)
    for i in 1:(rd ÷ 2)
        inv[i] = 1.0 / cfg.rope_theta^(2 * (i - 1) / rd)
    end
    return inv
end

# cos/sin tables (R, S, head_dim) for integer (0-based) positions (R, S)
function rope_tables(cfg::TimestralConfig, ::Type{T}, positions::Matrix{Int}) where T
    inv = rope_inv_freq(cfg)
    hd = head_dim(cfg)
    half = hd ÷ 2
    R, S = size(positions)
    cosT = Array{T}(undef, R, S, hd)
    sinT = Array{T}(undef, R, S, hd)
    emb = Vector{Float64}(undef, hd)
    for r in 1:R, s in 1:S
        pos = positions[r, s]
        @inbounds for i in 1:half
            emb[i] = inv[i] * pos
        end
        for i in 1:half
            emb[half + i] = emb[i]
        end
        for d in 1:hd
            cosT[r, s, d] = cos(emb[d])
            sinT[r, s, d] = sin(emb[d])
        end
    end
    return cosT, sinT
end

#=====================================================================================
  Attention
=====================================================================================#
function _softmax_rows!(S::Matrix{T}) where T
    n = size(S, 1)
    for i in 1:n
        m = S[i, 1]
        for j in 2:size(S, 2)
            S[i, j] > m && (m = S[i, j])
        end
        acc = zero(T)
        for j in 1:size(S, 2)
            e = exp(S[i, j] - m)
            S[i, j] = e
            acc += e
        end
        for j in 1:size(S, 2)
            S[i, j] /= acc
        end
    end
end

function attention(a::Attention{T}, x::Array{T,3},
                  time_mask::Union{Array{Bool,3},Nothing},
                  variate_mask::Union{Array{Bool,3},Nothing},
                  cosT::Array{T,3}, sinT::Array{T,3}) where T
    if a.is_time
        return _attn_over_slots(a, x, time_mask, cosT, sinT, true)
    else
        xp = permutedims(x, (2, 1, 3))          # (slots, rows, D)
        out = _attn_over_slots(a, xp, variate_mask, cosT, sinT, false)
        return permutedims(out, (2, 1, 3))
    end
end

# Batch axis 1, sequence axis 2. `mask` is (batch, seq, seq) for time attention
# or (seq, batch, batch) for variate attention (per-slot masks).
function _attn_over_slots(a::Attention{T}, x::Array{T,3},
                          mask::Union{Array{Bool,3},Nothing},
                          cosT::Array{T,3}, sinT::Array{T,3}, apply_rope::Bool) where T
    B, S, D = size(x)
    H, hd = a.num_heads, a.head_dim
    y = linear3(a.wQKV, x)                     # (B, S, 3D)
    q = Array{T}(undef, B, S, H, hd)
    k = Array{T}(undef, B, S, H, hd)
    v = Array{T}(undef, B, S, H, hd)
    for b in 1:B, s in 1:S, d in 1:hd, h in 1:H
        @inbounds q[b, s, h, d] = y[b, s, (d - 1) * H + h]
        @inbounds k[b, s, h, d] = y[b, s, D + (d - 1) * H + h]
        @inbounds v[b, s, h, d] = y[b, s, 2D + (d - 1) * H + h]
    end
    if apply_rope
        half = hd ÷ 2
        orig = Vector{T}(undef, hd)
        for b in 1:B, s in 1:S, h in 1:H
            @inbounds for d in 1:hd orig[d] = q[b, s, h, d] end
            for d in 1:hd
                rh = d <= half ? -orig[d + half] : orig[d - half]
                q[b, s, h, d] = orig[d] * cosT[b, s, d] + rh * sinT[b, s, d]
            end
        end
        for b in 1:B, s in 1:S, h in 1:H
            @inbounds for d in 1:hd orig[d] = k[b, s, h, d] end
            for d in 1:hd
                rh = d <= half ? -orig[d + half] : orig[d - half]
                k[b, s, h, d] = orig[d] * cosT[b, s, d] + rh * sinT[b, s, d]
            end
        end
    end
    out = Array{T}(undef, B, S, H, hd)
    scores = Matrix{T}(undef, S, S)
    scale = T(1 / sqrt(hd))
    for b in 1:B, h in 1:H
        for i in 1:S, j in 1:S
            acc = zero(T)
            @inbounds @simd for d in 1:hd
                acc += q[b, i, h, d] * k[b, j, h, d]
            end
            allowed = if mask === nothing
                apply_rope ? j <= i : true
            else
                mask[b, i, j]   # time: (rows, slots, slots); variate: (slots, rows, rows)
            end
            scores[i, j] = allowed ? acc * scale : -T(Inf)
        end
        _softmax_rows!(scores)
        for i in 1:S, d in 1:hd
            acc = zero(T)
            @inbounds for j in 1:S
                p = scores[i, j]
                p != 0 && (acc += p * v[b, j, h, d])
            end
            out[b, i, h, d] = acc
        end
    end
    merged = Array{T}(undef, B, S, D)
    for b in 1:B, s in 1:S, h in 1:H, d in 1:hd
        @inbounds merged[b, s, (h - 1) * hd + d] = out[b, s, h, d]
    end
    if a.gate !== nothing
        g = linear3(a.gate, x)
        for i in eachindex(merged)
            merged[i] *= T(1) / (T(1) + exp(-g[i]))
        end
    end
    return linear3(a.wO, merged)
end

#=====================================================================================
  Transformer stack
=====================================================================================#
function transformer_forward(m::TimestralModel{T}, x::Array{T,3},
                             time_mask::Union{Array{Bool,3},Nothing},
                             variate_mask::Union{Array{Bool,3},Nothing},
                             positions::Matrix{Int}) where T
    cosT, sinT = rope_tables(m.config, T, positions)
    for blk in m.blocks
        for j in 1:length(blk.attns)
            nx = rmsnorm(blk.attn_norms[j], x)
            ax = attention(blk.attns[j], nx, time_mask, variate_mask, cosT, sinT)
            x = x .+ ax
        end
        x = x .+ (blk.mlp)(rmsnorm(blk.mlp_norm, x))
    end
    return rmsnorm(m.out_norm, x)
end

#=====================================================================================
  Scaler (causal per-patch normalization, asinh, advancing frames)
=====================================================================================#
# data, obs: (R, T). Statistics accumulate in Float64, cast to T like the reference.
function causal_stats(data::Matrix{T}, obs::Matrix{Bool}, minimum_scale, ::Type{T}) where T
    R, Tn = size(data)
    means = Matrix{T}(undef, R, Tn)
    scales = Matrix{T}(undef, R, Tn)
    for r in 1:R
        cumw = 0.0; cumx = 0.0; m2 = 0.0
        prev_mean = 0.0
        for t in 1:Tn
            x = Float64(data[r, t])
            w = obs[r, t] ? 1.0 : 0.0
            cumw += w
            cumx += w * x
            den = max(cumw, 1.0)
            mean = cumx / den
            m2 += (x - prev_mean) * (x - mean) * w
            var = max(m2 / max(den - 1.0, 1.0), 0.0)  # guard tiny fp-negative M2
            means[r, t] = T(mean)
            scales[r, t] = T(max(sqrt(var), minimum_scale))
            prev_mean = mean
        end
    end
    return means, scales
end

# per-step (loc, scale) broadcast over each patch
function causal_patch_loc_scale(data::Matrix{T}, obs::Matrix{Bool}, ps::Int,
                                minimum_scale, ::Type{T}) where T
    R, Tn = size(data)
    size(data, 2) % ps == 0 || error("length $Tn is not a multiple of patch_size $ps")
    means, scales = causal_stats(data, obs, minimum_scale, T)
    loc = Matrix{T}(undef, R, Tn)
    scale = Matrix{T}(undef, R, Tn)
    for r in 1:R, p in 1:(Tn ÷ ps)
        @inbounds for d in 1:ps
            loc[r, (p - 1) * ps + d] = means[r, p * ps]
            scale[r, (p - 1) * ps + d] = scales[r, p * ps]
        end
    end
    return loc, scale
end

function apply_affine_asinh_masked(data::Matrix{T}, loc::Matrix{T}, scale::Matrix{T},
                                   obs::Matrix{Bool}, use_asinh::Bool) where T
    out = similar(data)
    for i in eachindex(data)
        if obs[i]
            z = (data[i] - loc[i]) / max(scale[i], eps(T))
            out[i] = use_asinh ? T(asinh(z)) : z
        else
            out[i] = zero(T)
        end
    end
    return out
end

# Map +/-inf to the nearest finite value along dim 3 of (R, S, Q); all-non-finite -> 0.
function clamp_nonfinite(vals::Array{T,3}) where T
    R, S, Q = size(vals)
    out = copy(vals)
    fin = Vector{T}(undef, Q)
    for r in 1:R, s in 1:S
        for q in 1:Q
            x = vals[r, s, q]
            if x == T(Inf) || x == T(-Inf)
                nfin = 0
                for q2 in 1:Q
                    y = vals[r, s, q2]
                    if isfinite(y); nfin += 1; fin[nfin] = y; end
                end
                if x == T(Inf)
                    out[r, s, q] = nfin > 0 ? maximum(@view fin[1:nfin]) : -T(Inf)
                else
                    out[r, s, q] = nfin > 0 ? minimum(@view fin[1:nfin]) : T(Inf)
                end
            end
        end
        for q in 1:Q
            if !isfinite(out[r, s, q]) && out[r, s, q] == out[r, s, q] # ±Inf only (NaN untouched)
                out[r, s, q] = zero(T)
            end
        end
    end
    return out
end

function invert_affine_asinh(x::Array{T,3}, loc::Matrix{T}, scale::Matrix{T},
                             use_asinh::Bool) where T
    out = similar(x)
    for r in 1:size(x, 1), s in 1:size(x, 2), q in 1:size(x, 3)
        z = use_asinh ? T(sinh(x[r, s, q])) : x[r, s, q]
        out[r, s, q] = z * scale[r, s] + loc[r, s]
    end
    return out
end

denormalize_quantiles(preds::Array{T,3}, loc::Matrix{T}, scale::Matrix{T}, use_asinh::Bool) where T =
    clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))

# ---- Welford combine / cumulative scan (Float64) ----
tiny64 = 2.2250738585072014e-308

function welford_combine(wa, ma, m2a, wb, mb, m2b)
    w = wa .+ wb
    ws = max.(w, tiny64)
    delta = mb .- ma
    mean = ma .+ delta .* wb ./ ws
    m2 = m2a .+ m2b .+ (delta .^ 2) .* (wa .* wb) ./ ws
    zero_w = w .== 0
    mean = ifelse.(zero_w, 0.0, mean)
    m2 = ifelse.(zero_w, 0.0, m2)
    return w, mean, m2
end

function cumulative_welford(w::Matrix{Float64}, mean::Matrix{Float64}, m2::Matrix{Float64})
    r, n = size(w)
    d = 1
    while d < n
        wl = zeros(Float64, r, n); ml = zeros(Float64, r, n); m2l = zeros(Float64, r, n)
        for i in (d+1):n
            @inbounds for j in 1:r
                wl[j, i] = w[j, i - d]
                ml[j, i] = mean[j, i - d]
                m2l[j, i] = m2[j, i - d]
            end
        end
        w, mean, m2 = welford_combine(wl, ml, m2l, w, mean, m2)
        d *= 2
    end
    return w, mean, m2
end

"""
Per-patch (loc, scale) of a decoded block in the advancing refine frame.
Returns (locs, scales), each (R, n_pred) Float64.
"""
function advancing_block_frames(context::Matrix{T}, obs::Matrix{Bool},
                                sinh_median::Array{Float64,3},
                                minimum_scale) where T
    ps = size(sinh_median, 3)
    R, Tc = size(context)
    n_ctx = Tc ÷ ps
    n_pred = size(sinh_median, 2)
    n_tot = n_ctx + n_pred
    values = zeros(Float64, R, n_tot, ps)
    weights = zeros(Float64, R, n_tot, ps)
    for r in 1:R, p in 1:n_ctx, d in 1:ps
        values[r, p, d] = context[r, (p - 1) * ps + d]
        weights[r, p, d] = obs[r, (p - 1) * ps + d] ? 1.0 : 0.0
    end
    cnt = Matrix{Float64}(undef, R, n_tot)
    mean = Matrix{Float64}(undef, R, n_tot)
    m2 = Matrix{Float64}(undef, R, n_tot)
    for r in 1:R, p in 1:n_tot
        c = 0.0; s = 0.0
        for d in 1:ps
            c += weights[r, p, d]
            s += values[r, p, d] * weights[r, p, d]
        end
        mu = s / max(c, 1.0)
        m = 0.0
        for d in 1:ps
            m += weights[r, p, d] * (values[r, p, d] - mu)^2
        end
        cnt[r, p] = c
        mean[r, p] = c == 0 ? 0.0 : mu
        m2[r, p] = c == 0 ? 0.0 : m
    end
    cnt, mean, m2 = cumulative_welford(cnt, mean, m2)
    # state after the last context patch (single column)
    cnt_v = cnt[:, n_ctx]
    mean_v = mean[:, n_ctx]
    m2_v = m2[:, n_ctx]
    offset = copy(mean_v)
    clip = sqrt(floatmax(Float32)) / 1e3
    locs = Matrix{Float64}(undef, R, n_pred)
    scales = Matrix{Float64}(undef, R, n_pred)
    ps_f = Float64(ps)
    for h in 1:n_pred
        var = max.(m2_v ./ max.(cnt_v .- 1.0, 1.0), 0.0)
        scale_h = max.(sqrt.(var), minimum_scale)
        @inbounds for r in 1:R
            locs[r, h] = mean_v[r]
            scales[r, h] = scale_h[r]
        end
        b_mean = Vector{Float64}(undef, R)
        b_m2 = Vector{Float64}(undef, R)
        for r in 1:R
            folded = Vector{Float64}(undef, ps)
            for d in 1:ps
                x = sinh_median[r, h, d] * scale_h[r] + mean_v[r] - offset[r]
                x = clamp(x, -clip, clip) + offset[r]
                folded[d] = x
            end
            mu = sum(folded) / ps
            ss = 0.0
            for d in 1:ps
                ss += (folded[d] - mu)^2
            end
            b_mean[r] = mu
            b_m2[r] = ss
        end
        cnt_v, mean_v, m2_v = welford_combine(cnt_v, mean_v, m2_v,
                                             fill(ps_f, R), b_mean, b_m2)
    end
    return locs, scales
end

# Non-causal per-row mean/std for known-future covariates: (loc, scale, is_binary).
function instance_norm_stats(values::Matrix{T}, obs::Matrix{Bool}, minimum_scale) where T
    R, Tn = size(values)
    loc = Vector{T}(undef, R)
    scale = Vector{T}(undef, R)
    is_binary = Vector{Bool}(undef, R)
    for r in 1:R
        n = 0.0
        for t in 1:Tn; obs[r, t] && (n += 1); end
        n = max(n, 1.0)
        s = 0.0
        for t in 1:Tn; obs[r, t] && (s += Float64(values[r, t])); end
        mu = s / n
        v = 0.0
        for t in 1:Tn
            obs[r, t] && (v += (Float64(values[r, t]) - mu)^2)
        end
        v /= n
        loc[r] = T(mu)
        scale[r] = T(max(sqrt(v), minimum_scale))
        zo = false
        anyobs = false
        for t in 1:Tn
            if obs[r, t]
                anyobs = true
                x = values[r, t]
                (x == zero(T) || x == one(T)) || (zo = true; break)
            end
        end
        is_binary[r] = !zo && anyobs
        if is_binary[r]
            loc[r] = zero(T)
            scale[r] = one(T)
        end
    end
    return loc, scale, is_binary
end

#=====================================================================================
  Masks (all Array{Bool}; True = attend)
=====================================================================================#
_boolarray(x) = convert(Array{Bool}, x)

function enc_key_validity(obs_per_patch::Matrix{Bool}, attend_missing::Bool)
    R, N = size(obs_per_patch)
    kv = Matrix{Bool}(undef, R, N)
    for r in 1:R
        started = false
        for p in 1:N
            started |= obs_per_patch[r, p]
            kv[r, p] = attend_missing ? (obs_per_patch[r, p] || started) : obs_per_patch[r, p]
        end
    end
    return kv
end

function enc_causal_mask(key_valid::Matrix{Bool})
    R, L = size(key_valid)
    m = Array{Bool}(undef, R, L, L)
    for r in 1:R, i in 1:L, j in 1:L
        m[r, i, j] = (j <= i && key_valid[r, j]) || i == j
    end
    return m
end

# (R, L+h, L+h) for one block without covariates
function forecast_time_mask(key_valid::Matrix{Bool}, h::Int)
    R, L = size(key_valid)
    enc = enc_causal_mask(key_valid)
    S = L + h
    m = zeros(Bool, R, S, S)
    for r in 1:R
        for i in 1:L, j in 1:L
            m[r, i, j] = enc[r, i, j]
        end
        for i in 1:h, j in 1:L                    # dec -> enc: valid keys
            m[r, L + i, j] = key_valid[r, j]
        end
        for i in 1:h, j in 1:h                    # dec -> dec: causal
            m[r, L + i, L + j] = j <= i
        end
    end
    return m
end

# (R, 2n, 2n) over a full [Enc ; Dec] window (the covariate path)
function two_stream_time_mask(min_context::Vector{Int}, n::Int, block_size::Int,
                              key_valid::Matrix{Bool},
                              bidirectional_rows::Union{AbstractVector{Bool},Nothing})
    R = size(key_valid, 1)
    S = 2n
    m = zeros(Bool, R, S, S)
    enc = enc_causal_mask(key_valid)
    for r in 1:R
        bidir = bidirectional_rows === nothing ? false : bidirectional_rows[r]
        # enc -> enc
        for i in 1:n, j in 1:n
            m[r, i, j] = bidir ? (key_valid[r, j] || i == j) : enc[r, i, j]
        end
        # dec patch p0 = p - 1 - min_context[r] in 0-based terms
        for i in n+1:S
            p0 = (i - n) - 1
            s = min_context[r]
            active = p0 >= s
            offset = max(p0 - s, 0)
            anchor = active ? s + block_size * (offset ÷ block_size) : 0
            for j in 1:n
                j0 = j - 1
                m[r, i, j] = active && j0 < anchor && key_valid[r, j]   # dec -> enc
            end
            for j in n+1:S
                q0 = (j - n) - 1
                s0 = min_context[r]
                qactive = q0 >= s0
                qoff = max(q0 - s0, 0)
                qanchor = qactive ? s0 + block_size * (qoff ÷ block_size) : 0
                same = anchor == qanchor
                m[r, i, j] = (same && j <= i && active && qactive) || (i == j)
            end
        end
    end
    return m
end

# (S, R, R) variate mask or nothing; S = N_enc + N_dec slots
function two_stream_variate_mask(group_ids::Vector{Int}, enc_key_valid::Matrix{Bool},
                                n_dec::Int,
                                target_rows::Union{AbstractVector{Bool},Nothing},
                                bidirectional_rows::Union{AbstractVector{Bool},Nothing})
    R = length(group_ids)
    R == 1 && return nothing
    n_enc = size(enc_key_valid, 2)
    S = n_enc + n_dec
    forbid = zeros(Bool, R, R)
    if target_rows !== nothing
        for i in 1:R, j in 1:R
            if !target_rows[i] && target_rows[j]
                forbid[i, j] = true
            end
        end
    end
    if bidirectional_rows !== nothing
        for i in 1:R, j in 1:R
            if bidirectional_rows[i] && !bidirectional_rows[j]
                forbid[i, j] = true
            end
        end
    end
    m = zeros(Bool, S, R, R)
    for s in 1:S, i in 1:R, j in 1:R
        key_ok = s <= n_enc ? enc_key_valid[j, s] : true
        m[s, i, j] = (group_ids[i] == group_ids[j] && key_ok && !forbid[i, j]) || i == j
    end
    return m
end

# first observed patch per group (0-based), then expanded to rows
function group_first_observed(obs_per_patch::Matrix{Bool}, group_ids::Vector{Int})
    groups = unique(group_ids)
    first_obs = zeros(Int, length(group_ids))
    gfirst = Dict{Int,Int}()
    for g in groups
        idx = findall(==(g), group_ids)
        fo = 0
        for p in 1:size(obs_per_patch, 2)
            if any(obs_per_patch[r, p] for r in idx); fo = p - 1; break; end
        end
        gfirst[g] = fo
    end
    for (i, g) in enumerate(group_ids)
        first_obs[i] = gfirst[g]
    end
    return first_obs
end

#=====================================================================================
  Model forward primitives
=====================================================================================#
# scaled (R, T) + observed -> (R, N, D)
function representation(m::TimestralModel{T}, scaled::Matrix{T}, observed::Matrix{Bool}) where T
    R, Tn = size(scaled)
    ps = m.config.patch_size
    N = Tn ÷ ps
    D = m.config.d_model
    x = Array{T}(undef, R, N, 2ps)
    for r in 1:R, p in 1:N, d in 1:ps
        x[r, p, d] = scaled[r, (p - 1) * ps + d]
        x[r, p, ps + d] = observed[r, (p - 1) * ps + d] ? zero(T) : one(T)
    end
    return (m.patch_proj)(x)
end

function forecast_tokens(m::TimestralModel{T}, rows::Int, n_patches::Int) where T
    D = m.config.d_model
    out = Array{T}(undef, rows, n_patches, D)
    for r in 1:rows, p in 1:n_patches, d in 1:D
        out[r, p, d] = m.forecast_query[d]
    end
    return out
end

# monotone (non-crossing) quantiles along the last axis of (R, N, ps, Q)
function monotone_quantiles(raw::Array{T,4}, levels::Vector{Float64},
                            width_factor::Float64) where T
    R, N, ps, Q = size(raw)
    m = something(findfirst(==(0.5), levels), 0)
    widths = zeros(Float64, Q)
    for i in m+1:Q
        widths[i] = levels[i] - levels[i-1]
    end
    for i in 1:m-1
        widths[i] = levels[i+1] - levels[i]
    end
    out = similar(raw)
    big = floatmax(T)
    wf = T(width_factor)
    for r in 1:R, n in 1:N, p in 1:ps
        median = raw[r, n, p, m]
        # levels above the median: median + cumsum of widths outward
        acc = zero(T)
        for i in (m + 1):Q
            acc += wf * T(widths[i]) * _softplus(raw[r, n, p, i])
            out[r, n, p, i] = clamp(median + acc, -big, big)
        end
        # levels below the median: median - cumsum of widths outward (from the median)
        acc = zero(T)
        for i in (m - 1):-1:1
            acc += wf * T(widths[i]) * _softplus(raw[r, n, p, i])
            out[r, n, p, i] = clamp(median - acc, -big, big)
        end
        out[r, n, p, m] = clamp(median, -big, big)
    end
    return out
end

# x (R, N, D) -> normalized quantiles (R, N*ps, Q)
function head_forward(m::TimestralModel{T}, x::Array{T,3}) where T
    cfg = m.config
    raw = (m.head)(x)                     # (R, N, ps*Q)
    R, N, _ = size(raw)
    ps, Q = cfg.patch_size, num_quantiles(cfg)
    preds = Array{T}(undef, R, N, ps, Q)
    for r in 1:R, n in 1:N, p in 1:ps, q in 1:Q
        preds[r, n, p, q] = raw[r, n, (p - 1) * Q + q]
    end
    preds = monotone_quantiles(preds, cfg.quantile_knots, cfg.quantile_width_factor)
    out = Array{T}(undef, R, N * ps, Q)
    for r in 1:R, n in 1:N, p in 1:ps, q in 1:Q
        out[r, (n - 1) * ps + p, q] = preds[r, n, p, q]
    end
    return out
end

# causal per-patch normalization of packed (R, T) rows -> (scaled, loc, scale)
function model_scale(m::TimestralModel{T}, inputs::Matrix{T}, observed::Matrix{Bool}) where T
    cfg = m.config
    loc, scale = causal_patch_loc_scale(inputs, observed, cfg.patch_size,
                                        cfg.minimum_scale, T)
    scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, cfg.use_asinh)
    return scaled, loc, scale
end

function known_cov_scaling(m::TimestralModel{T}, inputs::Matrix{T}, observed::Matrix{Bool},
                           scaled::Matrix{T}, loc::Matrix{T}, scale::Matrix{T},
                           known_rows::AbstractVector{Bool}) where T
    any(known_rows) || return scaled, loc, scale
    cfg = m.config
    loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, cfg.minimum_scale)
    # whole-row normalized, no asinh on unobserved/indicator rows
    kscaled = similar(scaled)
    for i in eachindex(inputs)
        if observed[i]
            z = (inputs[i] - loc_c[(i - 1) % size(inputs, 1) + 1]) /
                max(scale_c[(i - 1) % size(inputs, 1) + 1], eps(T))
            kscaled[i] = z
        else
            kscaled[i] = zero(T)
        end
    end
    if cfg.use_asinh
        for i in eachindex(inputs)
            r = (i - 1) % size(inputs, 1) + 1
            if observed[i] && !is_binary[r]
                kscaled[i] = T(asinh(kscaled[i]))
            end
        end
    end
    kscaled2 = Matrix{T}(undef, size(scaled))
    loc2 = Matrix{T}(undef, size(loc))
    scale2 = Matrix{T}(undef, size(scale))
    for r in 1:size(inputs, 1), t in 1:size(inputs, 2)
        if known_rows[r]
            kscaled2[r, t] = kscaled[r, t]; loc2[r, t] = loc_c[r]; scale2[r, t] = scale_c[r]
        else
            kscaled2[r, t] = scaled[r, t]; loc2[r, t] = loc[r, t]; scale2[r, t] = scale[r, t]
        end
    end
    return kscaled2, loc2, scale2
end

# transformer over [Enc ; Dec], then the head on the decoder tokens
function run_two_stream(m::TimestralModel{T}, x_enc::Array{T,3}, x_dec::Array{T,3},
                       time_mask::Array{Bool,3}, variate_mask::Union{Array{Bool,3},Nothing},
                       positions::Matrix{Int}) where T
    x = cat(x_enc, x_dec; dims = 2)
    h = transformer_forward(m, x, time_mask, variate_mask, positions)
    enc_len = size(x_enc, 2)
    return head_forward(m, h[:, enc_len+1:end, :])
end

function obs_per_patch(observed::Matrix{Bool}, ps::Int)
    R, Tn = size(observed)
    Tn % ps == 0 || error("length $Tn not a multiple of patch size $ps")
    opp = Matrix{Bool}(undef, R, Tn ÷ ps)
    for r in 1:R, p in 1:(Tn ÷ ps)
        v = false
        for d in 1:ps
            v |= observed[r, (p - 1) * ps + d]
        end
        opp[r, p] = v
    end
    return opp
end

# one block of horizon_patches after a packed (R, T) context -> raw (R, h*ps, Q)
function predict_block(m::TimestralModel{T}, inputs::Matrix{T}, observed::Matrix{Bool},
                       group_ids::Vector{Int}, horizon_patches::Int,
                       role_ids::Union{Vector{Int},Nothing},
                       known_future::Union{Matrix{T},Nothing},
                       known_future_mask::Union{Matrix{Bool},Nothing}) where T
    all_targets = role_ids === nothing || all(==(ROLE_TARGET), role_ids)
    if all_targets || !m.config.covariate_enabled
        return _predict_context_only(m, inputs, observed, group_ids, horizon_patches)
    end
    R = size(inputs, 1)
    ps = m.config.patch_size
    horizon_len = horizon_patches * ps
    kf = known_future === nothing ? zeros(T, R, horizon_len) : known_future
    kfm = known_future_mask === nothing ? zeros(Bool, R, horizon_len) : known_future_mask
    full_inputs = hcat(inputs, kf)
    full_observed = hcat(observed, kfm)
    return _predict_window(m, full_inputs, full_observed, group_ids, horizon_patches, role_ids)
end

function _predict_context_only(m::TimestralModel{T}, inputs::Matrix{T},
                               observed::Matrix{Bool}, group_ids::Vector{Int},
                               horizon_patches::Int) where T
    cfg = m.config
    ps = cfg.patch_size
    R, Tn = size(inputs)
    L = Tn ÷ ps
    h = horizon_patches
    scaled, loc, scale = model_scale(m, inputs, observed)
    x_enc = representation(m, scaled, observed)
    x_dec = forecast_tokens(m, R, h)
    opp = obs_per_patch(observed, ps)
    kv = enc_key_validity(opp, cfg.attend_missing_patches)
    tm = forecast_time_mask(kv, h)
    vm = two_stream_variate_mask(group_ids, kv, h, nothing, nothing)
    first_obs = group_first_observed(opp, group_ids)
    positions = Matrix{Int}(undef, R, L + h)
    for r in 1:R, s in 1:(L + h)
        positions[r, s] = (s - 1) - first_obs[r]
    end
    preds = run_two_stream(m, x_enc, x_dec, tm, vm, positions)   # (R, h*ps, Q)
    anchor_loc = Matrix{T}(undef, R, size(preds, 2))
    anchor_scale = Matrix{T}(undef, R, size(preds, 2))
    for r in 1:R, s in 1:size(preds, 2)
        anchor_loc[r, s] = loc[r, (L - 1) * ps + 1]
        anchor_scale[r, s] = scale[r, (L - 1) * ps + 1]
    end
    return denormalize_quantiles(preds, anchor_loc, anchor_scale, cfg.use_asinh)
end

function _predict_window(m::TimestralModel{T}, full_inputs::Matrix{T},
                         full_observed::Matrix{Bool}, group_ids::Vector{Int},
                         horizon_patches::Int, role_ids::Vector{Int}) where T
    cfg = m.config
    ps = cfg.patch_size
    R, W = size(full_inputs)
    W % ps == 0 || error("window length $W not a multiple of patch size $ps")
    h = horizon_patches
    1 <= h <= cfg.block_size || error("decode block $h must be in [1, $(cfg.block_size)] with covariates")
    n = W ÷ ps
    ll = n - h
    ll >= 1 || error("context must have at least one patch")
    target_rows = role_ids .== ROLE_TARGET
    known_rows = role_ids .== ROLE_KNOWN_COV
    scaled, loc, scale = model_scale(m, full_inputs, full_observed)
    scaled, loc, scale = known_cov_scaling(m, full_inputs, full_observed, scaled,
                                          loc, scale, known_rows)
    x_enc = representation(m, scaled, full_observed)             # (R, n, D)
    x_dec = forecast_tokens(m, R, n)                             # (R, n, D)
    opp = obs_per_patch(full_observed, ps)
    kv = enc_key_validity(opp, cfg.attend_missing_patches)
    s_rows = fill(ll, R)
    tm = two_stream_time_mask(s_rows, n, cfg.block_size, kv, known_rows)
    vm = two_stream_variate_mask(group_ids, kv, n, target_rows, known_rows)
    first_obs = group_first_observed(opp, group_ids)
    positions = Matrix{Int}(undef, R, 2n)
    for r in 1:R, s in 1:n
        positions[r, s] = (s - 1) - first_obs[r]
        positions[r, n + s] = positions[r, s]
    end
    preds = run_two_stream(m, x_enc, x_dec, tm, vm, positions)   # (R, n*ps, Q)
    # keep the horizon patches
    horizon_preds = Array{T}(undef, R, h * ps, size(preds, 3))
    for r in 1:R, p in (ll + 1):n, d in 1:ps, q in 1:size(preds, 3)
        horizon_preds[r, (p - ll - 1) * ps + d, q] = preds[r, (p - 1) * ps + d, q]
    end
    anchor_loc = Matrix{T}(undef, R, size(horizon_preds, 2))
    anchor_scale = Matrix{T}(undef, R, size(horizon_preds, 2))
    for r in 1:R, s in 1:size(horizon_preds, 2)
        anchor_loc[r, s] = loc[r, (ll - 1) * ps + 1]
        anchor_scale[r, s] = scale[r, (ll - 1) * ps + 1]
    end
    return denormalize_quantiles(horizon_preds, anchor_loc, anchor_scale, cfg.use_asinh)
end

#=====================================================================================
  Categorical target encoding (per-series smoothed mean)
=====================================================================================#
function _slots(codes::AbstractVector{Float64}, categories::Vector{Float64},
                nan_slot::Int, unseen::Union{Int,Nothing})
    n = length(codes)
    slots = Vector{Int}(undef, n)
    C = length(categories)
    for i in 1:n
        c = codes[i]
        if isnan(c)
            slots[i] = nan_slot
        else
            idx = searchsortedfirst(categories, isnan(c) ? 0.0 : c)
            idx = min(idx, max(C, 1))
            known = C > 0 && categories[idx] == c
            slots[i] = known ? idx : (unseen === nothing ? 0 : unseen)
        end
    end
    return slots
end

"""
    target_encode(target, past_codes, [future_codes]; smooth = 1.0)

Per-series smoothed mean-target encoding. NaN codes form a category of their
own; codes never seen in the context resolve to the series mean.
Returns `(encoded_past, encoded_future)`; `encoded_future` is `nothing` when no
future codes are given. Values are Float64 (cast to the model dtype downstream).
"""
function target_encode(target::AbstractVector{Float64}, past_codes::AbstractVector{Float64},
                       future_codes::Union{AbstractVector{Float64},Nothing} = nothing;
                       smooth::Float64 = 1.0)
    length(target) == length(past_codes) ||
        error("target and codes must align")
    Tn = length(target)
    observed = [isfinite(t) for t in target]
    tvals = [observed[i] ? target[i] : 0.0 for i in 1:Tn]
    item_count = sum(observed)
    item_mean = item_count > 0 ? sum(tvals) / item_count : 0.0
    categories = Float64[]
    for c in past_codes
        isfinite(c) && !(c in categories) && push!(categories, c)
    end
    sort!(categories)
    C = length(categories)
    nan_slot = C + 1
    past_slots = _slots(past_codes, categories, nan_slot, nothing)
    sums = zeros(Float64, nan_slot)
    counts = zeros(Float64, nan_slot)
    for i in 1:Tn
        s = past_slots[i]
        if s > 0
            sums[s] += tvals[i] * observed[i]
            counts[s] += observed[i] ? 1.0 : 0.0
        end
    end
    lookup = Vector{Float64}(undef, nan_slot + 1)
    for c in 1:nan_slot
        lookup[c] = (smooth * item_mean + sums[c]) / (smooth + counts[c])
    end
    lookup[nan_slot + 1] = item_mean
    enc_past = [lookup[past_slots[i]] for i in 1:Tn]
    if future_codes === nothing
        return enc_past, nothing
    end
    fut_slots = _slots(future_codes, categories, nan_slot, nan_slot + 1)
    enc_future = [lookup[fut_slots[i]] for i in 1:length(future_codes)]
    return enc_past, enc_future
end

#=====================================================================================
  Forecaster
=====================================================================================#
struct Forecast{T}
    quantiles::Array{T,4}      # (B, V, H, Q), sorted along Q
    levels::Vector{Float64}
end

quantile_levels(f::Forecast) = f.levels
median(f::Forecast) = f.quantiles[:, :, :, something(findfirst(==(0.5), f.levels), 0)]

"""
    quantile_at(f, level)

`(B, V, H)` forecast at one of the model's quantile levels.
"""
function quantile_at(f::Forecast, level::Real)
    idx = findfirst(==(Float64(level)), f.levels)
    idx === nothing && error("level $level is not one of the model's quantile levels")
    return f.quantiles[:, :, :, idx]
end

struct TimestralForecaster{T}
    model::TimestralModel{T}
    config::TimestralConfig
    max_context::Int
    per_shot_horizon_patches::Int
    quantile_real_cap_k::Float64
    levels::Vector{Float64}
    median_index::Int
end

function TimestralForecaster(model::TimestralModel{T}, max_context::Union{Nothing,Integer},
                            per_shot::Integer, cap_k::Real) where T
    cfg = model.config
    per_shot >= 1 || error("per_shot_horizon_patches must be >= 1")
    return TimestralForecaster{T}(model, cfg,
        max_context === nothing ? cfg.context_length : Int(max_context),
        min(Int(per_shot), cfg.block_size), Float64(cap_k),
        cfg.quantile_knots, median_index(cfg))
end

# ---------- HF Hub download ----------
function _download_checkpoint(repo::AbstractString; revision::AbstractString = "main")
    base = "https://huggingface.co/$(repo)/resolve/$(revision)"
    dir = joinpath(homedir(), ".cache", "timestral-jl", replace(repo, "/" => "__"))
    mkpath(dir)
    for f in ("config.json", "model.safetensors")
        dest = joinpath(dir, f)
        isfile(dest) || Downloads.download("$base/$f", dest)
    end
    return dir
end

"""
    from_pretrained(path_or_repo; dtype = Float32, max_context = nothing,
                    per_shot_horizon_patches = 32, quantile_real_cap_k = 1e4)

Load an exported Timestral checkpoint (`config.json` + `model.safetensors`)
from a local directory or a Hugging Face Hub repo id (downloaded and cached).
`dtype` is `Float32` (matches the reference pipeline) or `Float64`.
"""
function from_pretrained(path_or_repo::AbstractString;
                         dtype::Type{<:AbstractFloat} = Float32,
                         max_context::Union{Nothing,Integer} = nothing,
                         per_shot_horizon_patches::Integer = 32,
                         quantile_real_cap_k::Real = 1e4,
                         revision::AbstractString = "main")
    dir = isdir(path_or_repo) ? String(path_or_repo) :
          _download_checkpoint(path_or_repo; revision = revision)
    cfg = config_from_dict(json_parse(read(joinpath(dir, "config.json"), String)))
    w = load_safetensors(joinpath(dir, "model.safetensors"), dtype)
    model = TimestralModel(cfg, w)
    return TimestralForecaster(model, max_context, per_shot_horizon_patches,
                               quantile_real_cap_k)
end

# ---------- input shaping ----------
function _as_context(fc::TimestralForecaster{T}, context, observation_mask) where T
    x = _to_array3(context)
    b, v, t = size(x)
    mask = _boolarray(isfinite.(x))
    if observation_mask !== nothing
        um = _to_array3(observation_mask)
        mask = _boolarray(mask .& _boolarray(reshape(um .!= 0, b, v, t)))
    end
    xf = Array{T}(undef, b, v, t)
    for i in eachindex(x)
        xf[i] = mask[i] ? T(Float64(x[i])) : zero(T)
    end
    return xf, mask
end

# promote Any-nested JSON values / any Real array to Array{Float64,3}
function _to_array3(x)
    a = x
    if a isa AbstractVector && !(a isa AbstractArray{<:Real})
        a = convert(Vector{Any}, a)
    end
    A = _float64_array(a)
    nd = ndims(A)
    if nd == 1
        return reshape(A, 1, 1, :)
    elseif nd == 2
        # (B, L) -> (B, 1, L)
        return reshape(A, size(A, 1), 1, size(A, 2))
    elseif nd == 3
        return A
    else
        error("expected a (T,), (B, T) or (B, V, T) array, got ndims=$nd")
    end
end

function _float64_array(a::AbstractArray{<:Real})
    return convert(Array{Float64}, a)
end
function _float64_array(a)
    # nested Any arrays (from JSON): NaN was encoded as null -> nothing
    flat = Float64[]
    shapes = Int[]
    function walk(node, depth)
        if node === nothing || node === missing
            push!(flat, NaN)
        elseif node isa Bool
            push!(flat, node ? 1.0 : 0.0)
        elseif node isa Real
            push!(flat, Float64(node))
        elseif node isa AbstractVector
            if length(shapes) < depth
                push!(shapes, length(node))
            elseif shapes[depth] != length(node)
                error("ragged nested array")
            end
            for c in node
                walk(c, depth + 1)
            end
        else
            error("unsupported array element $(typeof(node))")
        end
    end
    walk(a, 1)
    isempty(shapes) && error("empty array")
    # flat is row-major (shapes...); reshape column-major with reversed dims
    # yields (last..first), so permute back to the original axis order.
    return permutedims(reshape(flat, reverse(shapes)...), reverse(1:length(shapes)))
end

# ---------- helpers ----------
function ffill_with_mask(inputs::Array{T,3}, mask::Array{Bool,3}) where T
    b, v, t = size(inputs)
    out = similar(inputs)
    for i in 1:b, j in 1:v
        lastobs = 0
        for k in 1:t
            mask[i, j, k] && (lastobs = k)
            out[i, j, k] = inputs[i, j, max(lastobs, 1)]
        end
    end
    return out
end

function pad_to_patch(inputs::Array{T,3}, mask::Array{Bool,3}, ps::Int) where T
    t = size(inputs, 3)
    if t % ps != 0
        padlen = ps - (t % ps)
        z = zeros(T, size(inputs, 1), size(inputs, 2), padlen)
        zm = zeros(Bool, size(inputs, 1), size(inputs, 2), padlen)
        return cat(z, inputs; dims = 3), cat(zm, mask; dims = 3)
    end
    return inputs, mask
end

replace_extreme(x::T, threshold, replacement = zero(T)) where T =
    isfinite(x) && abs(x) < threshold ? x : replacement

function nonnegative_context(inputs::Array{T,3}, mask::Array{Bool,3}) where T
    b, v, t = size(inputs)
    out = Matrix{Bool}(undef, b, v)
    for i in 1:b, j in 1:v
        anyobs = false
        ok = true
        for k in 1:t
            if mask[i, j, k]
                anyobs = true
                inputs[i, j, k] >= 0 || (ok = false; break)
            end
        end
        out[i, j] = anyobs && ok
    end
    return out
end

# ---------- categorical encoding (forecaster semantics) ----------
# target to encode against: (B, T), NaN outside the encoding window
function _encoding_target(fc::TimestralForecaster, inputs::Array{T,3},
                          mask::Array{Bool,3}) where T
    size(inputs, 2) == 1 || return nothing
    b, _, t = size(inputs)
    ff = ffill_with_mask(inputs, mask)
    out = Matrix{Float64}(undef, b, t)
    for i in 1:b
        started = false
        for k in 1:t
            started |= mask[i, 1, k]
            in_window = (k - 1) >= t - fc.max_context
            out[i, k] = (started && in_window) ? Float64(ff[i, 1, k]) : NaN
        end
    end
    return out
end

function _encode_categorical(fc::TimestralForecaster{T},
                             codes::Union{Array{Float64,3},Nothing},
                             target::Union{Matrix{Float64},Nothing}) where T
    (codes === nothing || target === nothing) && return codes
    b, c, L = size(codes)
    t = size(target, 2)
    out = Array{Float64}(undef, b, c, L)
    for i in 1:b, j in 1:c
        past = @views codes[i, j, 1:t]
        fut = L > t ? (@views codes[i, j, t+1:L]) : nothing
        enc_past, enc_fut = target_encode(target[i, :], past, fut)
        for k in 1:t
            out[i, j, k] = (isnan(target[i, k]) && isnan(codes[i, j, k])) ? NaN : enc_past[k]
        end
        if enc_fut !== nothing
            for k in 1:(L - t)
                out[i, j, t + k] = enc_fut[k]
            end
        end
    end
    return out
end

function _as_covariates(values, batch::Int, length_::Int, name::String, ::Type{T}) where T
    values === nothing && return nothing
    A = _float64_array(values)
    if ndims(A) == 2
        A = reshape(A, size(A, 1), 1, size(A, 2))
    end
    ndims(A) == 3 || error("$name must be (B, C, L)")
    size(A, 1) == batch || error("$name must have batch $batch, got $(size(A,1))")
    size(A, 3) == length_ || error("$name must have length $length_, got $(size(A,3))")
    size(A, 2) > 0 || return nothing
    out = Array{T}(undef, size(A))
    anyinf = false
    for i in eachindex(A)
        out[i] = T(A[i])
        isinf(A[i]) && (anyinf = true)
    end
    anyinf && error("$name must hold finite codes or NaN; got an infinite value")
    return out
end

function _as_covariates_f64(values, batch::Int, length_::Int, name::String)
    v = _as_covariates(values, batch, length_, name, Float64)
    return v  # already Float64
end

"""
    encode_categorical(fc, context; prediction_length, observation_mask = nothing,
                       categorical_past_covariates = nothing,
                       categorical_known_covariates = nothing)

Compute the target-encoded covariate rows exactly as `forecast` would pass them
to the model: `(encoded_past, encoded_known)`, each `(B, C, L)` or `nothing`.
"""
function encode_categorical(fc::TimestralForecaster, context; prediction_length::Int,
                            observation_mask = nothing,
                            categorical_past_covariates = nothing,
                            categorical_known_covariates = nothing)
    inputs, mask = _as_context(fc, context, observation_mask)
    b = size(inputs, 1)
    t = size(inputs, 3)
    cat_past = _as_covariates_f64(categorical_past_covariates, b, t,
                                  "categorical_past_covariates")
    cat_known = _as_covariates_f64(categorical_known_covariates, b, t + prediction_length,
                                   "categorical_known_covariates")
    target = _encoding_target(fc, inputs, mask)
    enc_past = _encode_categorical(fc, cat_past, target)
    enc_known = _encode_categorical(fc, cat_known, target)
    return enc_past, enc_known
end

# ---------- assembly ----------
function _assemble_rows(fc::TimestralForecaster{T}, inputs::Array{T,3},
                        mask::Array{Bool,3}, prediction_length::Int,
                        past::Union{Array{T,3},Nothing},
                        known::Union{Array{T,3},Nothing}) where T
    ps = fc.config.patch_size
    b, n_tgt, ctx0 = size(inputs)
    horizon = cld(prediction_length, ps) * ps

    rows_list = Vector{Array{T,3}}()
    mask_list = Vector{Array{Bool,3}}()
    roles = Int[]

    push!(rows_list, ffill_with_mask(inputs, mask))
    push!(mask_list, mask)
    append!(roles, fill(ROLE_TARGET, n_tgt))

    n_past = 0
    if past !== nothing
        n_past = size(past, 2)
        push!(mask_list, _boolarray(.!isnan.(past)))
        pz = copy(past)
        for i in eachindex(pz); isnan(pz[i]) && (pz[i] = zero(T)); end
        push!(rows_list, pz)
        append!(roles, fill(ROLE_PAST_COV, n_past))
    end

    n_known = 0
    known_future_vals = known_future_obs = nothing
    if known !== nothing
        n_known = size(known, 2)
        kc_ctx = known[:, :, 1:ctx0]
        kc_fut = known[:, :, ctx0+1:end]
        size(kc_ctx, 3) == ctx0 ||
            error("known covariates must cover the context plus the horizon")
        push!(mask_list, _boolarray(.!isnan.(kc_ctx)))
        kz = copy(kc_ctx)
        for i in eachindex(kz); isnan(kz[i]) && (kz[i] = zero(T)); end
        push!(rows_list, kz)
        append!(roles, fill(ROLE_KNOWN_COV, n_known))
        known_future_obs = _boolarray(.!isnan.(kc_fut))
        known_future_vals = copy(kc_fut)
        for i in eachindex(known_future_vals)
            isnan(known_future_vals[i]) && (known_future_vals[i] = zero(T))
        end
    end

    rows = reduce((a, c) -> cat(a, c; dims = 2), rows_list)
    mrows = reduce((a, c) -> cat(a, c; dims = 2), mask_list)
    if size(rows, 3) > fc.max_context
        rows = rows[:, :, end - fc.max_context + 1:end]
        mrows = mrows[:, :, end - fc.max_context + 1:end]
    end
    rows, mrows = pad_to_patch(rows, mrows, ps)
    v_all = size(rows, 2)
    if n_past == 0 && n_known == 0
        return rows, mrows, nothing, nothing, nothing, n_tgt
    end

    role_ids = repeat(roles, b)
    known_future = known_future_mask = nothing
    if n_known > 0
        k0 = n_tgt + n_past
        plen = size(known_future_vals, 3)
        kf = zeros(T, b, v_all, horizon)
        kfm = zeros(Bool, b, v_all, horizon)
        for i in 1:b, j in 1:n_known, k in 1:plen
            kf[i, k0 + j, k] = known_future_vals[i, j, k]
            kfm[i, k0 + j, k] = known_future_obs[i, j, k]
        end
        known_future = reshape(permutedims(kf, (2, 1, 3)), b * v_all, horizon)
        known_future_mask = reshape(permutedims(kfm, (2, 1, 3)), b * v_all, horizon)
    end
    return rows, mrows, role_ids, known_future, known_future_mask, n_tgt
end

# ---------- block-autoregressive decode ----------
function _refine_block(fc::TimestralForecaster{T}, quantiles::Array{T,3},
                       context::Matrix{T}, mask::Matrix{Bool},
                       target_rows::Union{AbstractVector{Bool},Nothing}) where T
    ps = fc.config.patch_size
    total = size(quantiles, 2)
    n_pred = total ÷ ps
    n_pred <= 1 && return quantiles
    _, loc, scale = model_scale(fc.model, context, mask)
    loc0 = loc[:, end]
    scale0 = scale[:, end]
    R = size(quantiles, 1)
    Q = size(quantiles, 3)
    mid = fc.median_index
    # median (R, n_pred, ps) in Float64, "asinh undone" frame of the anchor
    sinh_median = Array{Float64}(undef, R, n_pred, ps)
    for r in 1:R, p in 1:n_pred, d in 1:ps
        m = quantiles[r, (p - 1) * ps + d, mid]
        sinh_median[r, p, d] = (Float64(m) - loc0[r]) / scale0[r]
    end
    loc_j, scale_j = advancing_block_frames(context, mask, sinh_median,
                                            fc.config.minimum_scale)
    # first horizon patch keeps the anchor frame
    locH = Matrix{T}(undef, R, total)
    scaleH = Matrix{T}(undef, R, total)
    for r in 1:R, h in 1:n_pred
        l0 = h == 1 ? loc0[r] : T(loc_j[r, h])
        s0 = h == 1 ? scale0[r] : T(scale_j[r, h])
        for d in 1:ps
            locH[r, (h - 1) * ps + d] = l0
            scaleH[r, (h - 1) * ps + d] = s0
        end
    end
    sinh_norm = Array{T}(undef, R, total, Q)
    for r in 1:R, s in 1:total, q in 1:Q
        sinh_norm[r, s, q] = (quantiles[r, s, q] - loc0[r]) / scale0[r]
    end
    refined = Array{T}(undef, R, total, Q)
    for r in 1:R, s in 1:total, q in 1:Q
        refined[r, s, q] = sinh_norm[r, s, q] * scaleH[r, s] + locH[r, s]
    end
    refined = clamp_nonfinite(refined)
    if target_rows !== nothing
        for r in 1:R
            if !target_rows[r]
                @inbounds for s in 1:total, q in 1:Q
                    refined[r, s, q] = quantiles[r, s, q]
                end
            end
        end
    end
    return refined
end

function _block_ar_decode(fc::TimestralForecaster{T}, inputs::Array{T,3},
                          observation_mask::Array{Bool,3}, prediction_length::Int,
                          role_ids::Union{Vector{Int},Nothing},
                          known_future::Union{Matrix{T},Nothing},
                          known_future_mask::Union{Matrix{Bool},Nothing}) where T
    ps = fc.config.patch_size
    b, v, ctx_len = size(inputs)
    # pack rows b-major: r = (b-1)*v + v'  (permute v to the front, then reshape)
    context = reshape(permutedims(inputs, (2, 1, 3)), b * v, ctx_len)
    mask = reshape(permutedims(observation_mask, (2, 1, 3)), b * v, ctx_len)
    group_ids = repeat(collect(1:b); inner = v)

    known_rows = past_rows = target_rows = nothing
    if role_ids !== nothing
        known_rows = role_ids .== ROLE_KNOWN_COV
        past_rows = (role_ids .!= ROLE_TARGET) .& .!known_rows
        target_rows = role_ids .== ROLE_TARGET
    end

    blocks = Vector{Array{T,3}}()
    remaining = cld(prediction_length, ps)
    offset = 0
    while remaining > 0
        block = min(fc.per_shot_horizon_patches, remaining)
        blen = block * ps
        kf_block = known_future === nothing ? nothing :
                   known_future[:, offset+1:offset+blen]
        kfm_block = known_future_mask === nothing ? nothing :
                    known_future_mask[:, offset+1:offset+blen]
        quantiles = predict_block(fc.model, context, mask, group_ids, block,
                                  role_ids, kf_block, kfm_block)
        quantiles = _refine_block(fc, quantiles, context, mask, target_rows)
        push!(blocks, quantiles)
        remaining -= block
        offset += blen
        remaining <= 0 && break
        # feedback: targets carry their median, known rows their real future,
        # past-only rows a blank
        feedback = Matrix{T}(undef, size(context, 1), blen)
        feedback_mask = zeros(Bool, size(context, 1), blen)
        for r in 1:size(context, 1), k in 1:blen
            m = quantiles[r, k, fc.median_index]
            feedback[r, k] = replace_extreme(m, 1e30)
            feedback_mask[r, k] = true
        end
        if role_ids !== nothing
            for r in 1:size(context, 1)
                if known_rows[r] && kf_block !== nothing
                    for k in 1:blen
                        feedback[r, k] = kf_block[r, k]
                        feedback_mask[r, k] = kfm_block[r, k]
                    end
                end
                if past_rows[r]
                    for k in 1:blen
                        feedback[r, k] = zero(T)
                        feedback_mask[r, k] = false
                    end
                end
            end
        end
        context = hcat(context, feedback)
        mask = hcat(mask, feedback_mask)
        if size(context, 2) > fc.max_context
            keep = max(ps, (fc.max_context ÷ ps) * ps)
            context = context[:, end-keep+1:end]
            mask = mask[:, end-keep+1:end]
        end
    end
    packed = reduce((a, c) -> cat(a, c; dims = 2), blocks)   # (R, total, Q)
    packed = packed[:, 1:prediction_length, :]
    out = Array{T}(undef, b, v, prediction_length, size(packed, 3))
    for i in 1:b, j in 1:v, h in 1:prediction_length, q in 1:size(packed, 3)
        out[i, j, h, q] = packed[(i - 1) * v + j, h, q]
    end
    return out
end

function _cap_quantiles(fc::TimestralForecaster{T}, quantiles::Array{T,4},
                        inputs::Array{T,3}, observation_mask::Array{Bool,3}) where T
    b, v, t = size(inputs)
    ctx_max = Matrix{T}(undef, b, v)
    ctx_min = Matrix{T}(undef, b, v)
    anchor = Matrix{T}(undef, b, v)
    k = fc.quantile_real_cap_k
    k <= 0 && return quantiles
    flat_in = reshape(permutedims(inputs, (2, 1, 3)), b * v, t)
    flat_mask = reshape(permutedims(observation_mask, (2, 1, 3)), b * v, t)
    loc, scale = causal_patch_loc_scale(flat_in, flat_mask, fc.config.patch_size,
                                        fc.config.minimum_scale, T)
    for i in 1:b, j in 1:v
        r = (i - 1) * v + j
        mx = -T(Inf); mn = T(Inf); anyobs = false
        for tt in 1:t
            if flat_mask[r, tt]
                anyobs = true
                flat_in[r, tt] > mx && (mx = flat_in[r, tt])
                flat_in[r, tt] < mn && (mn = flat_in[r, tt])
            end
        end
        ctx_max[i, j] = anyobs ? mx : zero(T)
        ctx_min[i, j] = anyobs ? mn : zero(T)
        anchor[i, j] = scale[r, end] * k
    end
    lo = ctx_min .- anchor
    hi = ctx_max .+ anchor
    out = similar(quantiles)
    for i in 1:b, j in 1:v, h in 1:size(quantiles, 3), q in 1:size(quantiles, 4)
        out[i, j, h, q] = clamp(quantiles[i, j, h, q], lo[i, j], hi[i, j])
    end
    return out
end

function _forecast_raw(fc::TimestralForecaster{T}, inputs::Array{T,3},
                       mask::Array{Bool,3}, prediction_length::Int,
                       past::Union{Array{T,3},Nothing},
                       known::Union{Array{T,3},Nothing}) where T
    rows, mrows, role_ids, known_future, known_future_mask, n_tgt =
        _assemble_rows(fc, inputs, mask, prediction_length, past, known)
    quantiles = _block_ar_decode(fc, rows, mrows, prediction_length, role_ids,
                                 known_future, known_future_mask)
    quantiles = quantiles[:, 1:n_tgt, :, :]
    if fc.quantile_real_cap_k > 0
        quantiles = _cap_quantiles(fc, quantiles, rows[:, 1:n_tgt, :], mrows[:, 1:n_tgt, :])
    end
    if any(isnan, quantiles)
        error("forecast produced NaN values; the input is degenerate")
    end
    return quantiles
end

_negate(x) = x === nothing ? nothing : -x

function _cat_rows(a, b)
    if a === nothing; return b; end
    if b === nothing; return a; end
    return cat(a, b; dims = 2)
end

"""
    forecast(fc, context; prediction_length,
             observation_mask = nothing, past_covariates = nothing,
             known_covariates = nothing,
             categorical_past_covariates = nothing,
             categorical_known_covariates = nothing,
             symmetric_averaging = true, clip_nonnegative = true)

Forecast `prediction_length` steps after `context`.

`context` is `(B, V, T)`, `(B, T)` or `(T,)`; NaN marks a missing value.
`past_covariates` are `(B, C, T)`, `known_covariates` `(B, C, T + prediction_length)`;
categorical covariates hold raw category codes (NaN = missing) and are
target-encoded per series. Returns a `Forecast` with `quantiles` `(B, V, H, Q)`.
"""
function forecast(fc::TimestralForecaster, context;
                  prediction_length::Int,
                  observation_mask = nothing,
                  past_covariates = nothing,
                  known_covariates = nothing,
                  categorical_past_covariates = nothing,
                  categorical_known_covariates = nothing,
                  symmetric_averaging::Bool = true,
                  clip_nonnegative::Bool = true)
    prediction_length >= 1 || error("prediction_length must be >= 1")
    inputs, mask = _as_context(fc, context, observation_mask)
    b = size(inputs, 1)
    t = size(inputs, 3)
    T = eltype(inputs)
    past = _as_covariates(past_covariates, b, t, "past_covariates", T)
    known = _as_covariates(known_covariates, b, t + prediction_length,
                           "known_covariates", T)
    cat_past = _as_covariates_f64(categorical_past_covariates, b, t,
                                  "categorical_past_covariates")
    cat_known = _as_covariates_f64(categorical_known_covariates, b, t + prediction_length,
                                   "categorical_known_covariates")
    past_rows, known_rows = past, known
    if cat_past !== nothing || cat_known !== nothing
        target = _encoding_target(fc, inputs, mask)
        enc_past = _encode_categorical(fc, cat_past, target)
        enc_known = _encode_categorical(fc, cat_known, target)
        if enc_past !== nothing
            enc_past = convert(Array{T,3}, enc_past)
        end
        if enc_known !== nothing
            enc_known = convert(Array{T,3}, enc_known)
        end
        past_rows = _cat_rows(past_rows, enc_past)
        known_rows = _cat_rows(known_rows, enc_known)
    end
    if known_rows !== nothing && !supports_known_covariates(fc.config)
        error("this checkpoint cannot condition on known-future covariates")
    end
    mir_past = _negate(past_rows)
    mir_known = _negate(known_rows)

    quantiles = _forecast_raw(fc, inputs, mask, prediction_length, past_rows, known_rows)
    if symmetric_averaging
        mirrored = _forecast_raw(fc, -inputs, mask, prediction_length, mir_past, mir_known)
        Q = size(quantiles, 4)
        q2 = similar(quantiles)
        for i in eachindex(quantiles)
            q2[i] = quantiles[i]
        end
        for i in 1:size(quantiles, 1), j in 1:size(quantiles, 2),
            h in 1:size(quantiles, 3), q in 1:Q
            quantiles[i, j, h, q] = T(0.5) * (q2[i, j, h, q] - mirrored[i, j, h, Q + 1 - q])
        end
    end
    if clip_nonnegative
        eligible = nonnegative_context(inputs, mask)
        for i in 1:size(quantiles, 1), j in 1:size(quantiles, 2)
            if eligible[i, j]
                for h in 1:size(quantiles, 3), q in 1:size(quantiles, 4)
                    quantiles[i, j, h, q] < zero(T) && (quantiles[i, j, h, q] = zero(T))
                end
            end
        end
    end
    return Forecast{T}(quantiles, fc.levels)
end

end # module
