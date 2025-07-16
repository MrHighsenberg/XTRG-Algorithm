module plotting
export plot_partition_function, plot_singular_values
using Plots

"""
    plot_partition_function(betas, Zs, Zs_analytical)

Plot the partition function Z(β) comparing XTRG algorithm results with analytical solution.

Parameters:
- `betas`: Vector of inverse temperatures
- `Zs`: Partition function values from XTRG algorithm  
- `Zs_analytical`: Analytical partition function values
"""
function plot_partition_function(betas, Zs, Zs_analytical)
    # Plot partition function against inverse temperature
    plot(betas, Zs, 
         label="XTRG Algorithm", 
         marker=:circle, 
         markersize=4,
         linewidth=2,
         xscale=:log10,
         yscale=:log10
         )

    plot!(betas, Zs_analytical, 
          label="Analytical Solution", 
          linestyle=:dash,
          linewidth=2,
          legend=:topleft,
          grid=true
          )

    xlabel!("Inverse temperature \$\\beta\$")
    ylabel!("Partition Function \$Z(\\beta)\$")
    title!("Partition Function \$Z(\\beta)\$ of the XY Model")

    # Display the plot
    display(current())
end

"""
    plot_singular_values(betas, sing_value_lists, site_index, n_sing_values; Nsteps=length(betas)-1)

Plot the first n_sing_values singular values at a specific site as a function of inverse temperature β.

Parameters:
- `betas`: Vector of inverse temperatures
- `sing_value_lists`: Singular values from XTRG algorithm
- `site_index`: Index of the site/bond to analyze
- `n_sing_values`: Number of singular values to plot (starting from the largest)
- `Nsteps`: Number of XTRG steps (default: length(betas)-1)
"""
function plot_singular_values(betas, sing_value_lists, site_index, n_sing_values; Nsteps=length(betas)-1)
    # Initialize storage for singular values
    site_sing_values = [Float64[] for _ in 1:n_sing_values]
    min_n = 2  # Start from 2 since we don't have singular values for beta_0
    
    # Extract chaing length
    L = length(sing_value_lists[2])

    # Extract singular values for each XTRG step
    for n in 2:(Nsteps + 1)
        available_sing_vals = length(sing_value_lists[n][site_index])
        tot_weigth = sum(sing_value_lists[n][site_index])
        for i in 1:n_sing_values
            if i <= available_sing_vals
                push!(site_sing_values[i], sing_value_lists[n][site_index][i]/tot_weigth)
            end
        end
    end
    
    # Create the plot
    p = plot(xscale=:log10, yscale=:log10)
    
    # Plot each singular value series
    for i in 1:n_sing_values
        if length(site_sing_values[i]) > 0
            plot!(p, betas[end - length(site_sing_values[i]) + 1:end], site_sing_values[i],
                label="$(i)$(i==1 ? "st" : i==2 ? "nd" : i==3 ? "rd" : "th") value",
                marker=:circle,
                markersize=4,
                linewidth=2)
        end
    end
    
    xlabel!("Inverse temperature \$\\beta\$")
    ylabel!("Singular values")
    title!("Singular Values at Bond $site_index - $(site_index + 1)")
    plot!(legend=:bottomright)
    
    # Display the plot
    display(current())
end

end