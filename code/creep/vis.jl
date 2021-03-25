using KitBase.Plots

a = randn(28, 28)
b = rand(28, 28)

contour(a)
contour!(b)