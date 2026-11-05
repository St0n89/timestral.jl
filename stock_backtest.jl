#!/usr/bin/env julia
# stock_backtest.jl — historic-snapshot backtest for Timestral.jl
#
# Pick a ticker, a date range you want to use as the (historic) context, a
# forecast horizon and a checkpoint. The script downloads daily closes from
# Yahoo Finance, runs Timestral on the context, and scores the forecast
# against the real data that followed.
#
# Usage:
#   julia stock_backtest.jl <TICKER> <START> <END> <H> [MODEL] [weights_dir]
#
#     TICKER  Yahoo Finance symbol: AAPL, MSFT, SAP.DE, ^GDAXI, ...
#     START   context start date (YYYY-MM-DD)
#     END     context end date (YYYY-MM-DD) — your "historic snapshot"
#     H       forecast horizon: trading days after END to predict (and verify)
#     MODEL   5m | 28m | 123m | 209m   (default: 5m)
#     weights_dir  optional explicit checkpoint directory
#
# Example:
#   julia stock_backtest.jl AAPL 2024-01-02 2026-08-31 20 5m
#
# Outputs a per-day table (actual vs median, 80% band), summary metrics
# (MAE / MAPE / RMSE, 80% interval coverage, directional accuracy vs the
# last observed close, last-value baseline) and a CSV with all 21 quantiles.

include(joinpath(@__DIR__, "Timestral.jl"))
using .Timestral
using Dates, Downloads, Printf

const _UA = Dict("User-Agent" => "Mozilla/5.0 (compatible; TimestralStockBacktest/1.0)")

# ---------- data ----------

function fetch_daily_closes(ticker::AbstractString, d1::Date, d2::Date)
    p1 = round(Int, Dates.datetime2unix(Dates.DateTime(d1)))
    p2 = round(Int, Dates.datetime2unix(Dates.DateTime(d2)))
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$(ticker)" *
          "?period1=$(p1)&period2=$(p2)&interval=1d"
    buf = IOBuffer()
    Downloads.download(url, buf; headers = _UA)
    chart = Timestral.json_parse(String(take!(buf)))["chart"]
    err = get(chart, "error", nothing)
    err !== nothing && error("Yahoo Finance error: $(err)")
    length(get(chart, "result", Any[])) >= 1 ||
        error("no data returned for '$ticker' (check the symbol)")
    res = chart["result"][1]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][1]["close"]
    dates = Date[]
    vals = Float64[]
    for i in 1:length(ts)
        c = closes[i]
        c === nothing && continue          # null close (holiday / halted) -> skip
        push!(dates, Date(Dates.unix2datetime(ts[i])))
        push!(vals, Float64(c))
    end
    return dates, vals
end

# ---------- checkpoint ----------

function ensure_weights(model::AbstractString, explicit::Union{AbstractString,Nothing})
    model = lowercase(String(model))
    model in ("5m", "28m", "123m", "209m") ||
        error("model must be one of 5m, 28m, 123m, 209m (got '$model')")
    dir = explicit !== nothing ? String(explicit) : "/home/user/timestral-$(model)"
    if !isfile(joinpath(dir, "model.safetensors")) ||
       !isfile(joinpath(dir, "config.json"))
        @info "checkpoint not found at $dir — downloading from Hugging Face"
        mkpath(dir)
        repo = "mistralai/Timestral-$(uppercase(model))"
        for f in ("config.json", "model.safetensors")
            Downloads.download("https://huggingface.co/$(repo)/resolve/main/$(f)",
                              joinpath(dir, f); headers = _UA)
        end
    end
    return dir
end

# ---------- main ----------

function main()
    nargs = length(ARGS)
    nargs >= 4 || error("usage: julia stock_backtest.jl TICKER START END H [MODEL] [weights_dir]")
    ticker = uppercase(ARGS[1])
    start_d = Date(ARGS[2])
    end_d   = Date(ARGS[3])
    H       = parse(Int, ARGS[4])
    model   = nargs >= 5 ? ARGS[5] : "5m"
    wdir    = nargs >= 6 ? ARGS[6] : nothing
    H >= 1 || error("H must be >= 1")

    # fetch enough calendar after END to cover H trading days (holidays etc.)
    fetch_end = end_d + Day(2H + 10)
    all_dates, all_vals = fetch_daily_closes(ticker, start_d, fetch_end)

    ctx_mask = all_dates .<= end_d
    ctx_dates = all_dates[ctx_mask]
    ctx_vals  = all_vals[ctx_mask]
    isempty(ctx_dates) && error("no data on/before $end_d for $ticker")
    length(ctx_vals) >= 33 ||
        error("context has only $(length(ctx_vals)) trading days; give a longer START..END range")

    fut_idx = findall(d -> d > end_d, all_dates)
    length(fut_idx) >= H || error("only $(length(fut_idx)) trading days of real data " *
        "available after $end_d (need $H) — pick an earlier END or smaller H")
    fut_dates = all_dates[fut_idx[1:H]]
    fut_vals  = all_vals[fut_idx[1:H]]

    # forecast from the historic snapshot (Float32, like the reference)
    fc = from_pretrained(ensure_weights(model, wdir); dtype = Float32)
    f = forecast(fc, [ctx_vals]; prediction_length = H,
                 symmetric_averaging = true, clip_nonnegative = true)
    levels = Timestral.quantile_levels(f)
    Q = Array{Float64}(f.quantiles)[1, 1, :, :]        # (H, 21) rows sorted by level
    iq(l) = something(findfirst(==(l), levels), 0)
    i10, i50, i90 = iq(0.1), iq(0.5), iq(0.9)
    med = Q[:, i50]

    # metrics
    last_ctx = ctx_vals[end]
    mae  = sum(abs.(med .- fut_vals)) / H
    mape = 100 * sum(abs.((med .- fut_vals) ./ fut_vals)) / H
    rmse = sqrt(sum((med .- fut_vals) .^ 2) / H)
    cov  = 100 * count(i -> Q[i, i10] <= fut_vals[i] <= Q[i, i90], 1:H) / H
    dhit = 0; dcount = 0
    for t in 1:H
        a = sign(fut_vals[t] - last_ctx); m = sign(med[t] - last_ctx)
        a == 0 && continue
        dcount += 1
        a == m && (dhit += 1)
    end
    dir_acc = 100 * dhit / max(dcount, 1)
    base_mae  = sum(abs.(last_ctx .- fut_vals)) / H
    base_mape = 100 * sum(abs.((last_ctx .- fut_vals) ./ fut_vals)) / H

    # report
    println("=" ^ 78)
    println("Timestral stock backtest — $ticker | model $model | dtype f32")
    println("context  $(ctx_dates[1]) .. $(ctx_dates[end])  ($(length(ctx_vals)) trading days, " *
            "last close $(@sprintf("%.2f", last_ctx)))")
    println("forecast $(fut_dates[1]) .. $(fut_dates[end])  ($H trading days, verified against real data)")
    println("=" ^ 78)
    @printf("%-12s %10s %10s %10s %10s %10s %10s  %s\n",
            "date", "actual", "median", "q10", "q90", "err%", "band", "dir")
    for t in 1:H
        errp = 100 * (med[t] - fut_vals[t]) / fut_vals[t]
        band = fut_vals[t] < Q[t, i10] ? "below" : fut_vals[t] > Q[t, i90] ? "above" : "in"
        a = sign(fut_vals[t] - last_ctx); m = sign(med[t] - last_ctx)
        dir = a == 0 ? "-" : a == m ? "ok" : "x"
        @printf("%-12s %10.2f %10.2f %10.2f %10.2f %+9.2f%% %10s  %s\n",
                fut_dates[t], fut_vals[t], med[t], Q[t, i10], Q[t, i90], errp, band, dir)
    end
    println("-" ^ 78)
    @printf("Timestral : MAE %.2f | MAPE %.2f%% | RMSE %.2f | 80%%-band %.0f%% | direction %.0f%%\n",
            mae, mape, rmse, cov, dir_acc)
    @printf("last value: MAE %.2f | MAPE %.2f%%\n", base_mae, base_mape)
    println("=" ^ 78)

    # CSV with all quantiles
    csv = "stock_backtest_$(ticker)_$(end_d)_H$H.csv"
    open(csv, "w") do io
        println(io, "date,actual," * join(["q$l" for l in levels], ","))
        for t in 1:H
            println(io, fut_dates[t], ",", fut_vals[t], ",",
                    join((@sprintf("%.6f", Q[t, q]) for q in 1:size(Q, 2)), ","))
        end
    end
    println("wrote $csv ($(size(Q, 2)) quantile columns)")
end

main()
