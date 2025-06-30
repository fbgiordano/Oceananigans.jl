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

#free_surface = ImplicitFreeSurface()
free_surface = ExplicitFreeSurface()
                      
##### 
##### Build boundary conditions
#####

using Oceananigans.BoundaryConditions: Open
import Oceananigans.BoundaryConditions: getbc, update_boundary_condition!

struct OrlanskiBoundary{U, U1, U2, C}
    uᴮ :: U
    u1 :: U1
    u2 :: U2
    c :: C
end

OrlanskiBoundaryCondition = BoundaryCondition{<:Open, <:OrlanskiBoundary}

uᵂ  = Field{Nothing, Nothing, Center}(grid)
uᴱ  = Field{Nothing, Nothing, Center}(grid)
#u₁ᵂ = Field{Nothing, Nothing, Center}(grid)
#u₁ᴱ = Field{Nothing, Nothing, Center}(grid)
u₁ᵂ = Field{Face, Nothing, Center}(grid)
u₁ᴱ = Field{Face, Nothing, Center}(grid)
u₂ᵂ = Field{Face, Nothing, Center}(grid)
u₂ᴱ = Field{Face, Nothing, Center}(grid)
cʷ = Field{Nothing, Nothing, Center}(grid)
cᵉ = Field{Nothing, Nothing, Center}(grid)
c0 = sqrt(9.80655 * grid.Lz) * 0.1minutes / 10kilometers
set!(cʷ, -c0)
set!(cᵉ, c0)
@info cʷ[1,1,1]

u_west = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ, u₂ᵂ, cʷ))
u_east = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ, u₂ᴱ, cᵉ))

@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) = bc.condition.uᴮ[1, j, k]

u_bcs = FieldBoundaryConditions(west=u_west, east=u_east)

@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁, u₂, Δt, c)
    j, k = @index(Global, NTuple)

    Δux = @inbounds (uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k])
    Δut = @inbounds (uⁿ⁺¹[2, j, k] - u₁[2, j, k])
    # Δut = @inbounds (uⁿ⁺¹[2, j, k] -   u₁[1, j, k])
    # c   = ifelse(Δux == 0, zero(grid), abs(Δut / Δux))
    #c = sqrt(9.80655 * grid.Lz) * Δt/Δxᶠᶜᶜ(1, j, k, grid)

    #@inbounds c[1, j, k] = c[1, j, k] * Δt / Δxᶠᶜᶜ(1, j, k, grid)
    #@info "West:"
    #@info uᴮ[1, j, k]
    #@info uⁿ⁺¹[2,j,k], uⁿ⁺¹[1,j,k]

    #@inbounds uᴮ[1, j, k] =   uᴮ[1, j, k] - c[1, j, k] * Δux ##kinda works
    #@inbounds uᴮ[1, j, k] =   (uᴮ[1, j, k] - c[1, j, k] * uⁿ⁺¹[2,j,k]) / (1 - c[1, j, k]) ##kinda works
    ##@inbounds c[1,j,k] = ifelse(u₁[3,j,k] - u₁[1,j,k] .!= 0, -2 * (uⁿ⁺¹[2,j,k] - u₁[2,j,k]) / (u₁[3,j,k] - u₁[1,j,k]), c[1,j,k])
    ##@inbounds uᴮ[1, j, k] = (uᴮ[1, j, k] - 0.5*c[1, j, k] * (4*uⁿ⁺¹[2,j,k] - uⁿ⁺¹[3,j,k])) / (1 - 1.5*c[1, j, k]) ##c update before?
    #@inbounds uᴮ[1, j, k] = (uᴮ[1, j, k]/Δt - c[1, j, k]/grid.Δxᶠᵃᵃ * uⁿ⁺¹[2,j,k]) / (1/Δt - c[1, j, k]/grid.Δxᶠᵃᵃ)
    
    ## -Orlanski
    #@inbounds uᴮ[1,j,k] = (1+c[1,j,k]) / (1-c[1,j,k]) * u₁[1,j,k] - 2*c[1,j,k] / (1-c[1,j,k]) * uⁿ⁺¹[2,j,k]
    #@inbounds ctmp = -0.5 * (uⁿ⁺¹[2,j,k] - u₂[2,j,k]) / (uⁿ⁺¹[2,j,k] + u₂[2,j,k] - u₁[3,j,k])
    #@inbounds c[1,j,k] = ifelse(isnan(ctmp), -1, ctmp) #c[1,j,k]
    ###...
    #@inbounds uᴮ[1,j,k] = 2*(u₂[2,j,k]*u₁[1,j,k] - 0.5*u₁[3,j,k]*u₁[1,j,k] + uⁿ⁺¹[2,j,k]^2 - uⁿ⁺¹[2,j,k]*u₂[2,j,k]) / (2*uⁿ⁺¹[2,j,k]-u₁[3,j,k])
    #@inbounds uᴮ[1,j,k] = 2*(u₂[2,j,k]*u₁[1,j,k] - 0.5*u₁[3,j,k]*uᴮ[1,j,k] + uⁿ⁺¹[2,j,k]^2 - uⁿ⁺¹[2,j,k]*u₂[2,j,k]) / (2*uⁿ⁺¹[2,j,k]-u₁[3,j,k])
    #@inbounds uᴮ[1,j,k] = ifelse(isnan(uᴮ[1,j,k]), 0, uᴮ[1,j,k])
    """
    ## -from ROMS
    #@inbounds g₁ = u₁[1,j,k] - u₁[1,j-1,k] #for 2D
    #@inbounds g₂ = u₁[2,j,k] - u₁[2,j-1,k] #for 2D
    @inbounds uₜ = u₁[2,j,k] - uⁿ⁺¹[2,j,k] ##-1, uⁿ⁺¹ → uᴮ
    @inbounds uₓ = uⁿ⁺¹[2,j,k] - uⁿ⁺¹[3,j,k] ##-1, uⁿ⁺¹ → uᴮ
    @inbounds uₜ = ifelse(uₜ * uₓ < 0.0, 0.0, uₜ)
    @inbounds cₓ = uₜ * uₓ
    @inbounds cff = max(uₓ^2, 1e-16)
    #@inbounds cₑ = min(cff, max(uₜ * uₑ, -cff)) #for 2D
    @inbounds uᴮ[1,j,k] = (cff * u₁[1,j,k] + cₓ * uⁿ⁺¹[2,j,k]) / (cff + cₓ) #
    """
    """
    ## -from Stevens
    #@inbounds g₁ = u₁[1,j,k] - u₁[1,j-1,k] #for 2D
    #@inbounds g₂ = u₁[2,j,k] - u₁[2,j-1,k] #for 2D
    @inbounds uₜ = uⁿ⁺¹[2,j,k] - u₁[2,j,k] ##-1, uⁿ⁺¹ → uᴮ
    @inbounds uₓ = uⁿ⁺¹[3,j,k] - uⁿ⁺¹[2,j,k] ##-1, uⁿ⁺¹ → uᴮ
    #@inbounds uₜ = ifelse(uₜ * uₓ < 0.0, 0.0, uₜ)
    @inbounds cₓ = uₜ/uₓ
    @inbounds cₓ = ifelse(cₓ < 0.0, 0.0, cₓ)
    @inbounds cₓ = ifelse(cₓ > 1.0, 1.0, cₓ)
    @inbounds cₓ = ifelse(isnan(cₓ), 1.0, cₓ)
    #@inbounds cₑ = min(cff, max(uₜ * uₑ, -cff)) #for 2D
    #@inbounds uᴮ[1,j,k] = u₁[1,j,k] + cₓ * (uⁿ⁺¹[2,j,k] - uⁿ⁺¹[1,j,k]) #
    @inbounds uᴮ[1,j,k] = (u₁[1,j,k] + cₓ * uⁿ⁺¹[2,j,k]) / (1 + cₓ)
    """

    ### -to avoid nans?
    """
    @inbounds uₜ = uⁿ⁺¹[2,j,k] - u₁[2,j,k]
    @inbounds uₓ = uⁿ⁺¹[3,j,k] - uⁿ⁺¹[2,j,k]
    @inbounds uₜ = ifelse(uₜ * uₓ < 0.0, 0.0, uₜ)
    eps = 1e-12
    @inbounds cₓ = uₜ * uₓ / max(eps , uₓ^2)
    @inbounds uᴮ[1,j,k] = (u₁[1,j,k] + cₓ * uⁿ⁺¹[2,j,k]) / (1 + cₓ)
    """
    #"""
    @inbounds uₜ = uⁿ⁺¹[2,j,k] - u₁[2,j,k]
    @inbounds uₓ = -uⁿ⁺¹[4,j,k] + 4*uⁿ⁺¹[3,j,k] - 3*uⁿ⁺¹[2,j,k]
    #@inbounds uₜ = u₁[1,j,k] - u₂[1,j,k] ### !!!
    #@inbounds uₓ = -u₁[4,j,k] + 4*u₁[3,j,k] - 3*u₁[2,j,k] ### !!!
    #@inbounds uₜ = ifelse(uₜ * uₓ < 0.0, 0.0, uₜ) #0.0 → -uₜ
    eps = 1e-4
    @inbounds cₓ = uₜ * uₓ / max(eps , uₓ^2)
    @inbounds cₓ = ifelse(cₓ < 0.0, 0.0, cₓ)
    #@inbounds cₓ = ifelse(cₓ > 1.0, 1.0, cₓ)
    @inbounds uᴮ[1,j,k] = (u₁[1,j,k] + cₓ * (4*uⁿ⁺¹[2,j,k] - uⁿ⁺¹[3,j,k])) / (1 + 3*cₓ) #u₁[1,j,k]
    #@inbounds uᴮ[1,j,k] = u₁[1,j,k] + cₓ * (-3*uⁿ⁺¹[1,j,k] + 4*uⁿ⁺¹[2,j,k] - uⁿ⁺¹[3,j,k])
    #"""
    @info "w:" cₓ
    ###

    @inbounds u₂ = u₁
    @inbounds u₁ = uⁿ⁺¹
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁, u₂, Δt, c)
    j, k = @index(Global, NTuple)
    Nx   = size(grid, 1)

    Δux = @inbounds (uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx-1, j, k])
    Δut = @inbounds (uⁿ⁺¹[Nx, j, k] - u₁[Nx, j, k])
    # Δut = @inbounds (uⁿ⁺¹[Nx, j, k] -   u₁[1, j, k])
    # c   = ifelse(Δux == 0, zero(grid), abs(Δut / Δux))
    #c = sqrt(9.80655 * grid.Lz) * Δt/Δxᶠᶜᶜ(Nx+1, j, k, grid)

    #@inbounds c[1, j, k] = c[1, j, k] * Δt / Δxᶠᶜᶜ(Nx+1, j, k, grid)
    #@info "East:"
    #@info uᴮ[1, j, k]
    #@info uⁿ⁺¹[Nx,j,k], uⁿ⁺¹[Nx+1,j,k]

    #@inbounds uᴮ[1, j, k] =   uᴮ[1,  j, k] - c[1, j, k] * Δux ##kinda works
    #@inbounds uᴮ[1, j, k] =   (uᴮ[1, j, k] + c[1, j, k] * uⁿ⁺¹[Nx,j,k]) / (1 + c[1, j, k]) ##kinda works
    ##@inbounds c[1,j,k] = ifelse(u₁[Nx-1,j,k] - u₁[Nx+1,j,k] .!= 0, -2 * (uⁿ⁺¹[Nx,j,k] - u₁[Nx,j,k]) / (u₁[Nx-1,j,k] - u₁[Nx+1,j,k]), c[1,j,k]) ##Nx-1 - Nx+1?
    ##@inbounds uᴮ[1, j, k] = (uᴮ[1, j, k] + 0.5*c[1, j, k] * (4*uⁿ⁺¹[Nx,j,k] - uⁿ⁺¹[Nx-1,j,k])) / (1 + 1.5*c[1, j, k]) ##c update before?
    #@inbounds uᴮ[1, j, k] = (uᴮ[1, j, k]/Δt + c[1, j, k]/grid.Δxᶠᵃᵃ * uⁿ⁺¹[Nx,j,k]) / (1/Δt - c[1, j, k]/grid.Δxᶠᵃᵃ)

    ## -Orlanski
    #@inbounds uᴮ[1,j,k] = (1-c[1,j,k]) / (1+c[1,j,k]) * u₁[Nx+1,j,k] + 2*c[1,j,k] / (1+c[1,j,k]) * uⁿ⁺¹[Nx,j,k]
    #@inbounds ctmp = 0.5 * (uⁿ⁺¹[Nx,j,k] - u₂[Nx,j,k]) / (uⁿ⁺¹[Nx,j,k] + u₂[Nx,j,k] - u₁[Nx-1,j,k])
    #@inbounds c[1,j,k] = ifelse(isnan(ctmp), 1, ctmp) #c[1,j,k]
    ###...
    #@inbounds uᴮ[1,j,k] = 2*(u₂[Nx,j,k]*u₁[Nx+1,j,k] - 0.5*u₁[Nx-1,j,k]*u₁[Nx+1,j,k] + uⁿ⁺¹[Nx,j,k]^2 - uⁿ⁺¹[Nx,j,k]*u₂[Nx,j,k]) / (2*uⁿ⁺¹[Nx,j,k]-u₁[Nx-1,j,k])
    #@inbounds uᴮ[1,j,k] = 2*(u₂[Nx,j,k]*u₁[Nx+1,j,k] - 0.5*u₁[Nx-1,j,k]*uᴮ[1,j,k] + uⁿ⁺¹[Nx,j,k]^2 - uⁿ⁺¹[Nx,j,k]*u₂[Nx,j,k]) / (2*uⁿ⁺¹[Nx,j,k]-u₁[Nx-1,j,k])
    #@inbounds uᴮ[1,j,k] = ifelse(isnan(uᴮ[1,j,k]), 0, uᴮ[1,j,k])
    #"""
    """
    ## -from ROMS
    #@inbounds g₁ = u₁[Nx,j,k] - u₁[Nx,j-1,k] #for 2D
    #@inbounds g₂ = u₁[Nx+1,j,k] - u₁[Nx+1,j-1,k] #for 2D
    @inbounds uₜ = u₁[Nx,j,k] - uⁿ⁺¹[Nx,j,k] ##+1
    @inbounds uₓ = uⁿ⁺¹[Nx,j,k] - uⁿ⁺¹[Nx-1,j,k] ##+1
    @inbounds uₜ = ifelse(uₜ * uₓ < 0.0, 0.0, uₜ)
    @inbounds cₓ = uₜ * uₓ
    @inbounds cff = max(uₓ^2, 1e-16)
    #@inbounds cₑ = min(cff, max(uₜ * uₑ, -cff)) #for 2D
    @inbounds uᴮ[1,j,k] = (cff * u₁[Nx+1,j,k] + cₓ * uⁿ⁺¹[Nx,j,k]) / (cff + cₓ) #
    """
    """
    ## -from Stevens
    #@inbounds g₁ = u₁[1,j,k] - u₁[1,j-1,k] #for 2D
    #@inbounds g₂ = u₁[2,j,k] - u₁[2,j-1,k] #for 2D
    @inbounds uₜ = uⁿ⁺¹[Nx,j,k] - u₁[Nx,j,k] ##
    @inbounds uₓ = uⁿ⁺¹[Nx,j,k] - uⁿ⁺¹[Nx-1,j,k] ## switch signs?
    #@inbounds uₜ = ifelse(uₜ * uₓ < 0.0, 0.0, uₜ)
    @inbounds cₓ = uₜ/uₓ
    @inbounds cₓ = ifelse(cₓ > 0.0, 0.0, cₓ)
    @inbounds cₓ = ifelse(cₓ < -1.0, -1.0, cₓ)
    @inbounds cₓ = ifelse(isnan(cₓ), -1.0, cₓ)
    #@inbounds cₑ = min(cff, max(uₜ * uₑ, -cff)) #for 2D
    #@inbounds uᴮ[1,j,k] = u₁[Nx+1,j,k] + cₓ * (uⁿ⁺¹[Nx+1,j,k] - uⁿ⁺¹[Nx,j,k]) #
    @inbounds uᴮ[1,j,k] = (u₁[Nx+1,j,k] - cₓ * uⁿ⁺¹[Nx,j,k]) / (1 - cₓ)
    """

    #@info cff, cₓ
    #"""

    ### -to avoid nans?
    """
    @inbounds uₜ = uⁿ⁺¹[Nx,j,k] - u₁[Nx,j,k]
    @inbounds uₓ = uⁿ⁺¹[Nx,j,k] - uⁿ⁺¹[Nx-1,j,k]
    @inbounds uₜ = ifelse(uₜ * uₓ > 0.0, 0.0, uₜ)
    eps = 1e-12
    @inbounds cₓ = uₜ * uₓ / max(eps , uₓ^2)
    @inbounds uᴮ[1,j,k] = (u₁[Nx+1,j,k] - cₓ * uⁿ⁺¹[Nx,j,k]) / (1 - cₓ)
    """
    #"""
    @inbounds uₜ = uⁿ⁺¹[Nx,j,k] - u₁[Nx,j,k]
    @inbounds uₓ = uⁿ⁺¹[Nx-2,j,k] - 4*uⁿ⁺¹[Nx-1,j,k] + 3*uⁿ⁺¹[Nx,j,k]
    #@inbounds uₜ = ifelse(uₜ * uₓ > 0.0, 0.0, uₜ) #0.0 → -uₜ
    eps = 1e-4
    @inbounds cₓ = uₜ * uₓ / max(eps , uₓ^2)
    @inbounds cₓ = ifelse(cₓ > 0.0, 0.0, cₓ)
    #@inbounds cₓ = ifelse(cₓ < -1.0, -1.0, cₓ)
    @inbounds uᴮ[1,j,k] = (u₁[Nx+1,j,k] - cₓ * (4*uⁿ⁺¹[Nx,j,k] - uⁿ⁺¹[Nx-1,j,k])) / (1 - 3*cₓ) #u₁[Nx+1,j,k]
    #@inbounds uᴮ[1,j,k] = u₁[Nx+1,j,k] - cₓ * (-3*uⁿ⁺¹[Nx+1,j,k] + 4*uⁿ⁺¹[Nx,j,k] - uⁿ⁺¹[Nx-1,j,k])
    #"""
    @info "e:" cₓ
    ###

    @inbounds u₂ = u₁
    @inbounds u₁ = uⁿ⁺¹
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:west}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u₂ = bc.condition.u2
    c = bc.condition.c
    u  = model.velocities.u
    grid = model.grid
    #Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    Δt = 0.1minutes
    
    launch!(architecture(grid), grid, :yz,  _update_west_bc, uᴮ, grid, u, u₁, u₂, Δt, c)
    
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:east}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u₂ = bc.condition.u2
    c = bc.condition.c
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    
    launch!(architecture(grid), grid, :yz,  _update_east_bc, uᴮ, grid, u, u₁, u₂, Δt, c)
    
    return nothing
end

function update_boundary_condition!(bcs::FieldBoundaryConditions, u, model)
    update_boundary_condition!(bcs.west, Val(:west), u, model)
    update_boundary_condition!(bcs.east, Val(:east), u, model)
    
    #impose_volume_conservation!(bcs.west, bcs.east, model)
    
    return nothing
end

using Oceananigans.Operators
#"""
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
#"""
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

simulation = Simulation(model, Δt=0.1minutes, stop_time=3days)

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

ylims!(axu, (-2e-1, 2e-1))
ylims!(axv, (-2e-1, 2e-1))
ylims!(axw, (-1e-5, 1e-5))
ylims!(axη, (-0.05, 0.15))


record(fig, "hydrostatic_open_boundaries.mp4", 1:Nt) do i 
    @info "doing iteration $i of $Nt"
    n[] = i
end

@info (time_ns() - wt) / 1e9
