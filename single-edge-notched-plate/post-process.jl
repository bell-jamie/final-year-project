using CSV
using PyPlot
using DataFrames

csv_file = "nl-coupled-recursive-dat/fullSolve.csv"
data = DataFrame(CSV.File(csv_file))

displacement = data[:, 1]
force = data[:, 2]

plot(displacement * 1e3, force)
xlabel("Displacement (mm)")
ylabel("Load (N)")
title("Force vs Displacement - Single Edge Notched Plate - Non-Linear Recursive")
display(gcf())