using CairoMakie, NipponColors
using KitBase.JLD2

dc = dict_color()
cd(@__DIR__)

@load "optim_record.jld2" his his1

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(log10.(his); color = dc["ro"], label = "Shakhov")
    axislegend()
    fig
end
