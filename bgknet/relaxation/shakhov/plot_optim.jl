using CairoMakie, NipponColors
using KitBase.JLD2

dc = dict_color()
cd(@__DIR__)

@load "optim_record.jld2" his his1

t = collect(1:length(his)) .* 30
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "t", ylabel = L"C", title = "")
    lines!(t, log10.(his ./ 500); color = dc["ro"], label = "Train")
    lines!(t, log10.(his1 ./ 500); color = dc["tsuyukusa"], label = "Test", linestyle=:dash)
    axislegend()
    fig
end
save("shakhov_train.pdf", fig)
