module plotting
export plot_partition_function, plot_singular_values, plot_Z_error
using Plots, Measures, Statistics, LinearAlgebra

"""
    plot_partition_function(betas, Zs, Zs_analytical)

Plots the partition function Z(beta) as obtained from the XTRG algorithm comparing it with the analytical solution.

Parameters:
- `betas`: Vector of inverse temperatures
- `Zs`: Partition function values from XTRG algorithm  
- `Zs_analytical`: Analytical partition function values
"""
function plot_partition_function(betas, Zs, Zs_analytical)
    # Plot partition function against inverse temperature
    p = plot(betas, Zs, 
         label="XTRG Algorithm", 
         marker=:circle, 
         markersize=4,
         linewidth=2,
         xscale=:log10,
         yscale=:log10,
         size=(800,500),
         dpi = 300,
         left_margin=5mm
         )

    plot!(betas, Zs_analytical, 
          label="Analytical Solution", 
          linestyle=:dash,
          linewidth=2,
          legend=:topleft,
          grid=true
          )

    xlabel!("Inverse Temperature \$\\beta\$")
    ylabel!("Partition Function \$Z(\\beta)\$")
    title!("Partition Function \$Z(\\beta)\$ of the XY Model")

    # Save the plot 
    dir = dirname("plots/partition_function_xy-model.png")
    if !isdir(dir)
        mkpath(dir)
    end
    savefig(p, "plots/partition_function_xy-model.png")

    # Display the plot
    display(p)
end

"""
    plot_Z_error(betas, Zs, Zs_analytical)

Plots the partition function relative error (Z_num - Z_ex)/Z_ex as a function of beta.

Parameters:
- `betas`: Vector of inverse temperatures
- `Zs`: Partition function values from XTRG algorithm  
- `Zs_analytical`: Analytical partition function values
"""
function plot_Z_error(betas, Zs, Zs_analytical)
    # Compute relative errors
    Zerrors = (Zs .- Zs_analytical) ./ Zs_analytical

    # Plot relative errors against inverse temperature
    p = plot(betas, abs.(Zerrors), 
         label="Relative Error |Z_XTRG - Z_exact|/Z_exact", 
         marker=:circle, 
         markersize=4,
         linewidth=2,
         xscale=:log10,
         yscale=:log10,
         size=(800,500),
         dpi = 300,
         left_margin=5mm
         )

    xlabel!("Inverse Temperature \$\\beta\$")
    ylabel!("Relative Error")
    title!("Partition Function Relative Error")

    # Save the plot 
    dir = dirname("plots/partition_function_error.png")
    if !isdir(dir)
        mkpath(dir)
    end
    savefig(p, "plots/partition_function_error.png")

    # Display the plot
    display(p)
end


"""
    plot_singular_values(betas, singvals, bond_index, n_singvals; Nsteps)

Plots the first n_singvals singular values at a specific site as a function of inverse temperature beta.

Parameters:
- `betas`: Vector of inverse temperatures
- `singvals`: Singular values from XTRG algorithm
- `bond_index`: Index of the bond to analyze
- `n_singvals`: Number of singular values to plot (starting from the largest)
- `Nsteps`: Number of XTRG steps to plot
"""
function plot_singular_values(betas, singvals, bond_index, n_singvals; Nsteps=length(betas)-1)
    # Initialize storage for singular values
    site_singvals = [Float64[] for _ in 1:n_singvals]

    # Extract singular values for each XTRG step
    for n in 2:(Nsteps + 1)
        available_singvals = length(singvals[n][bond_index])
        # Divide by total weight states are unnormalized
        tot_weight = sum(singvals[n][bond_index])
        for i in 1:n_singvals
            if i <= available_singvals
                push!(site_singvals[i], singvals[n][bond_index][i] / tot_weight)
            end
        end
    end
    
    # Create doubly logarithmic plot
    p = plot(xscale=:log10, yscale=:log10, size=(800,500), dpi=300, left_margin=5mm)
    
    # Plot each singular value series
    for i in 1:n_singvals
        if length(site_singvals[i]) > 0
            plot!(p, betas[end - length(site_singvals[i]) + 1:end], site_singvals[i],
                label="$(i)$(i==1 ? "st" : i==2 ? "nd" : i==3 ? "rd" : "th") value",
                marker=:circle,
                markersize=4,
                linewidth=2)
        end
    end
    
    xlabel!("Inverse Temperature \$\\beta\$")
    ylabel!("Singular Values")
    title!("Singular Values at Bond ($bond_index, $(bond_index + 1))")
    plot!(legend=:bottomright)

    # Save the plot
    dir = dirname("plots/singular_values_bond_($bond_index, $(bond_index + 1)).png")
    if !isdir(dir)
        mkpath(dir)
    end
    savefig(p, "plots/singular_values_bond_($bond_index, $(bond_index + 1)).png")  

    # Display the plot
    display(p)
end

end