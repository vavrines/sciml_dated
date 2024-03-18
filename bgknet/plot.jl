using CairoMakie, NipponColors
using KitBase.JLD2

dc = dict_color()
cd(@__DIR__)

@load "optim_record.jld2" his his1 his2 his3

t = collect(1:length(his)) .* 30
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "t", ylabel = L"C", yscale = log10)
    lines!(t, (his ./ 1200); color = dc["ro"], label = "Train")
    lines!(t, (his1 ./ 1200); color = dc["tsuyukusa"], label = "Test", linestyle=:dash)
    lines!(t, (his2 ./ 500);)
    lines!(t, (his3 ./ 500);)
    axislegend()
    fig
end
#save("shakhov_train.pdf", fig)
