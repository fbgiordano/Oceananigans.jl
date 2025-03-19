using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Grids: architecture
using Oceananigans.Models

using KernelAbstractions: @kernel, @index

wt = time_ns()

z = MutableVerticalDiscretization((-10, 0))

grid = RectilinearGrid(size = (50, 10),     
                          x = (0, 500kilometers),
                          # y = (0, 500kilometers),
                          topology = (Bounded, Flat, Bounded),
                          z = z)

free_surface = ImplicitFreeSurface() 
                      
##### 
##### Build boundary conditions
#####

using Oceananigans.BoundaryConditions: Open
import Oceananigans.BoundaryConditions: getbc, update_boundary_condition!

struct OrlanskiBoundary{U, U1}
    uᴮ :: U
    u1 :: U1
end

OrlanskiBoundaryCondition = BoundaryCondition{<:Open, <:OrlanskiBoundary}

uᵂ  = Field{Nothing, Nothing, Center}(grid)
uᴱ  = Field{Nothing, Nothing, Center}(grid)
u₁ᵂ = Field{Nothing, Nothing, Center}(grid)
u₁ᴱ = Field{Nothing, Nothing, Center}(grid)

u_west = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ))
u_east = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ))

@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) = bc.condition.uᴮ[1, j, k]

u_bcs = FieldBoundaryConditions(west=u_west, east=u_east)

@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
    j, k = @index(Global, NTuple)

    Δux = @inbounds (uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k]) 
    # Δut = @inbounds (uⁿ⁺¹[2, j, k] -   u₁[1, j, k])
    # c   = ifelse(Δux == 0, zero(grid), abs(Δut / Δux))

    c = sqrt(9.80665 * grid.Lz) * Δt / Δxᶠᶜᶜ(1, j, k, grid)

    @inbounds uᴮ[1, j, k] =   uᴮ[1, j, k] + c * Δux
    # @inbounds u₁[1, j, k] = uⁿ⁺¹[2, j, k]
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
    j, k = @index(Global, NTuple)
    Nx   = size(grid, 1)

    Δux = @inbounds (uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx-1, j, k])
    # Δut = @inbounds (uⁿ⁺¹[Nx, j, k] -   u₁[1, j, k])
    # c   = ifelse(Δux == 0, zero(grid), abs(Δut / Δux))

    c = sqrt(9.80665 * grid.Lz) * Δt / Δxᶠᶜᶜ(Nx+1, j, k, grid)

    @inbounds uᴮ[1, j, k] =   uᴮ[1,  j, k] - c * Δux 
    # @inbounds u₁[1, j, k] = uⁿ⁺¹[Nx, j, k]
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:west}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    
    launch!(architecture(grid), grid, :yz,  _update_west_bc, uᴮ, grid, u, u₁, Δt)
    
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:east}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    
    launch!(architecture(grid), grid, :yz,  _update_east_bc, uᴮ, grid, u, u₁, Δt)
    
    return nothing
end

function update_boundary_condition!(bcs::FieldBoundaryConditions, u, model)
    update_boundary_condition!(bcs.west, Val(:west), u, model)
    update_boundary_condition!(bcs.east, Val(:east), u, model)
    
    impose_volume_conservation!(bcs.west, bcs.east, model)
    
    return nothing
end

using Oceananigans.Operators

@kernel function _impose_volume_conservation!(uw, ue, grid)
    j = @index(Global, Linear)

    Uc = 0
    for k in 1:grid.Nz
        Uc += @inbounds uw[1, j, k] * Δzᶠᶜᶜ(1, j, k, grid) + ue[1, j, k] * Δzᶠᶜᶜ(grid.Nx+1, j, k, grid)
    end

    for k in 1:grid.Nz
        @inbounds ue[1, j, k] = ue[1, j, k] - Uc / grid.Lz 
        @inbounds uw[1, j, k] = uw[1, j, k] - Uc / grid.Lz 
    end
end

impose_volume_conservation!(u_west, u_east, model) = nothing

function impose_volume_conservation!(u_west::OrlanskiBoundaryCondition, u_east::OrlanskiBoundaryCondition, model)
    grid = model.grid
    launch!(architecture(grid), grid, (grid.Ny, ),  _impose_volume_conservation!, u_west.condition.uᴮ, u_east.condition.uᴮ, grid)
    return nothing
end

#####
##### Build the model
#####

model = HydrostaticFreeSurfaceModel(; grid,
                                      free_surface,
                                    #   vertical_coordinate = ZStar(),
                                      boundary_conditions = (; u=u_bcs))

#####
##### Set initial conditions
#####

Rx = 250kilometers
Ry = 250kilometers
σ  = 50kilometers

gaussian_bump(x, z) = 0.1 * exp(-((x - Rx)^2 / σ^2))

set!(model, η = gaussian_bump)

simulation = Simulation(model, Δt=0.1minutes, stop_time=1days)

#####
##### Attach an output writer and run!
#####

u, v, w = model.velocities
η = model.free_surface.η

simulation.output_writers[:total_velocities] = JLD2Writer(model, (; u, v, w),
                                                          schedule = TimeInterval(10minutes),
                                                          filename = "hydrostatic_open_boundaries.jld2",
                                                          overwrite_existing = true)

simulation.output_writers[:free_surface] = JLD2Writer(model, (; η),
                                                      schedule = TimeInterval(10minutes),
                                                      filename = "hydrostatic_open_boundaries_free_surface.jld2",
                                                      overwrite_existing = true)

run!(simulation)

#####
##### Visualize the output
#####

using GLMakie

u = FieldTimeSeries("hydrostatic_open_boundaries.jld2", "u")
v = FieldTimeSeries("hydrostatic_open_boundaries.jld2", "v")
w = FieldTimeSeries("hydrostatic_open_boundaries.jld2", "w")
η = FieldTimeSeries("hydrostatic_open_boundaries_free_surface.jld2", "η")

Nt = length(u.times)

fig = Figure(size = (1000, 500))
axu = Axis(fig[1, 1], title = "u-velocity")
axv = Axis(fig[1, 2], title = "v-velocity")
axw = Axis(fig[2, 1], title = "w-velocity")
axη = Axis(fig[2, 2], title = "free surface")

n = Observable(1)

un = @lift(interior(u[$n], :, 1, 10))
vn = @lift(interior(v[$n], :, 1, 10))
wn = @lift(interior(w[$n], :, 1, 11))
ηn = @lift(interior(η[$n], :, 1, 1))

lines!(axu, un)
lines!(axv, vn)
lines!(axw, wn)
lines!(axη, ηn)

ylims!(axw, (-1e-5, 1e-5))

record(fig, "hydrostatic_open_boundaries.mp4", 1:Nt) do i 
    @info "doing iteration $i of $Nt"
    n[] = i
end

@info (time_ns() - wt) / 1e9
