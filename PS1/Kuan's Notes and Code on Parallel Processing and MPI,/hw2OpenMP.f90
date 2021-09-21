    !This program solves a household's decision problem using OpenMP.



    module parameters
    implicit none

    real*8, parameter ::  beta=0.98D+00
    real*8, parameter ::  zero=0.0d00, one=1.0d00
    real*8, parameter ::  alb=0.01d00, aub=16.0d00
    real*8, parameter ::  y=4.0d00   ! income
    real*8, parameter ::  r=0.03d00  ! interest rate
    real*8, parameter ::  tol=1.0D-3 ! tolerance of VFI
    integer, parameter :: na=600

    end module


    module global
    use parameters
    implicit none

    ! grid for state
    real*8, dimension(na) :: agrid

    !objects for VFI
    real*8 :: supnorm
    real*8 :: inc
    integer :: iter
    integer :: ia


    real*8, dimension(na) ::   v0, v1
    integer, dimension(na) ::  opta_ind_temp
    real*8, dimension(na)  ::  value
    real*8, dimension(na)  ::  opta,optc
    integer, dimension(na) ::  opta_ind

    end module

    
    PROGRAM example

    use parameters
    use global
    use omp_lib ! This is the OpenMP library comes with your compiler
    
    implicit none

    real:: total, etime, dist
    real, dimension(2):: elapsed

    real*8, dimension(na)  :: vtmp, cons
    logical, dimension(na) :: pos_cons
    integer, dimension(1)  :: adummy

    ! create a grid for assett
    agrid(1)=alb
    agrid(na)=aub

    inc=(aub-alb)/(na-1)
    do ia=2,na-1
        agrid(ia)=agrid(1)+inc*(ia-1)
    end do

    ! set up intital guess of the value function
    v0=zero

    !start VFI
    supnorm=one
    iter=0

    DO WHILE (supnorm>tol) ! while loop starts
        iter=iter+1

        !$omp parallel default(shared) private(ia, cons, pos_cons, vtmp, adummy)
        !$omp do
        ! loop over each state
        do ia=1,na
            cons=y+(one+r)*agrid(ia)-agrid
            pos_cons=(cons>zero)
            where (pos_cons==.true.)
                vtmp=log(cons)+beta*v0
            end where
            adummy=maxloc(vtmp,pos_cons)
            opta_ind_temp(ia)=adummy(1)
            v1(ia)=vtmp(opta_ind_temp(ia))
        end do
        !$omp end do
        !$omp end parallel

        supnorm=maxval(abs(v1-v0))
        print*, 'Iteration =',iter, 'Supnorm',supnorm
        v0 = v1

    END DO ! while loop ends


    total=etime(elapsed)


    PRINT*,'--------------------------------------------------'
    PRINT*,'total time elpased =',total, 'seconds'
    PRINT*,'--------------------------------------------------'

    ! Extract solutions
    do ia=1,na
        opta(ia)=agrid(opta_ind_temp(ia))
        optc(ia)=y+(one+r)*agrid(ia)-opta(ia)
    end do

    
    END PROGRAM