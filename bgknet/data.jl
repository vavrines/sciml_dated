using KitBase, Plots
using KitBase.CSV, KitBase.JLD2

cd(@__DIR__)

# ------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------






# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

file = "sample.csv"
f = open(file)
data = []
for line in eachline(f)
    a = split(line, ",")
    b = [parse(Float64, a[i]) for i = 2:length(a)]
    push!(data, b)
end
pdfs = data[3:end]

unum = length(data[1])
vs = VSpace1D(-5, 5, unum, Float32.(data[1]), zeros(Float32, unum), Float32.(data[2]))

plot(vs.u, pdfs[1000])
