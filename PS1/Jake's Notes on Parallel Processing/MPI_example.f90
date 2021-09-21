! Jake Zhao
! Department of Economics
! University of Wisconsin-Madison
! Econ 899: Computation of Heterogeneous Agent Models (Dean Corbae)
! Fall 2012

! Compile with the command "mpif90 MPI_example.f90 -o mpi -O2"
! Run with the command "mpirun -np 8 mpi"

PROGRAM main
IMPLICIT NONE
INCLUDE "mpif.h"

	INTEGER                           :: ierror               ! Error message from MPI subroutines
	INTEGER                           :: proc_id              ! Identification number of each processor
	INTEGER                           :: num_proc             ! Total number of processors
	INTEGER                           :: master_id = 0        ! Master processor identification number
	INTEGER                           :: ii, jj               ! Counters
	INTEGER, PARAMETER                :: mat_dim = 8000       ! Size of the matrix along one dimension
	REAL                              :: time_start, time_end ! Timing variables
	REAL                              :: time_1, time_2       ! Serial and parallel timings respectively
	REAL, DIMENSION(mat_dim,mat_dim)  :: matrix               ! Matrix to compute the sum
	REAL, DIMENSION(mat_dim)          :: sum_1, sum_2         ! Sum of each column
	REAL, DIMENSION(:,:), ALLOCATABLE :: temp_matrix          ! Temporary matrix for each processor
	REAL, DIMENSION(:),   ALLOCATABLE :: temp_sum             ! Temporary sum for each processor

	CALL MPI_init(ierror)                                ! Starts MPI processes
	CALL MPI_comm_rank(MPI_comm_world, proc_id, ierror)  ! Find the process ranks for each process, proc_id = 0, 1, 2, ...
	CALL MPI_comm_size(MPI_comm_world, num_proc, ierror) ! Find number of processes requested, determined when using mpirun

	PRINT*, 'Processor id', proc_id

	IF(proc_id == master_id)THEN
		CALL random_number(matrix) ! Fills matrix with uniform random numbers in [0,1]
		CALL cpu_time(time_start)

		sum_1 = 0
		DO jj=1,mat_dim
			DO ii=1,mat_dim
				sum_1(jj) = sum_1(jj) + matrix(ii,jj) ! Sum of each column
			END DO
		END DO

		CALL cpu_time(time_end)
		time_1 = time_end-time_start
		WRITE(*,'(a)'),		 ' $$$$$$$$$$$$$$$$$$$$$$$$'
		WRITE(*,'(a,2x,f9.4)'), ' Time elapsed ', time_1
		WRITE(*,'(a)'),		 ' $$$$$$$$$$$$$$$$$$$$$$$$'
	END IF

	IF(mat_dim/num_proc == REAL(mat_dim)/REAL(num_proc))THEN
		CALL MPI_barrier(MPI_comm_world, ierror)
		IF (proc_id == master_id) THEN
			CALL cpu_time(time_start)
		END IF

		ALLOCATE(temp_sum(mat_dim/num_proc))            ! Allocate a vector for each processor
		ALLOCATE(temp_matrix(mat_dim,mat_dim/num_proc)) ! Allocate a matrix for each processor

		! Scatters the random matrix to each processor and memory block
		! The MPI scatter command arguments are as follows: the matrix to be scattered, the number of elements to be scattered, the element type, the scattered matrix for each processor to compute,
		! the number of elements to be scattered, the element type, the processor that does the scattering, the scope, the return value of a possible error
		CALL MPI_scatter(matrix(:,proc_id*mat_dim/num_proc+1:(proc_id+1)*mat_dim/num_proc), mat_dim**2/num_proc, &
			& MPI_real, temp_matrix, mat_dim**2/num_proc, MPI_real, master_id, MPI_comm_world, ierror)

		temp_sum = 0
		DO jj=1,mat_dim/num_proc
			DO ii=1,mat_dim
				temp_sum(jj) = temp_sum(jj) + temp_matrix(ii,jj) ! Sum of each column for the scattered matrices
			END DO
		END DO

		CALL MPI_barrier(MPI_comm_world, ierror)
		! Gathers the computed sums from each processor and memory block
		! The MPI gather command arguments are as follows: the vectors to be gathered, the number of elements to be gathered, the element type, the gathered location,
		! the number of elements to be gathered, the element type, the processor that does the gathering, the scope, the return value of a possible error
		CALL MPI_gather(temp_sum, mat_dim/num_proc, MPI_real, sum_2(proc_id*mat_dim/num_proc+1:(proc_id+1)*mat_dim/num_proc), &
			& mat_dim/num_proc, MPI_real, master_id, MPI_comm_world, ierror)
		CALL MPI_finalize(ierror)

		IF (proc_id == master_id) THEN
			CALL cpu_time(time_end)
			time_2 = time_end-time_start
			WRITE(*,'(a)'),		 ' $$$$$$$$$$$$$$$$$$$$$$$$'
			WRITE(*,'(a,2x,f9.4)'), ' Time elapsed ', time_2
			WRITE(*,'(a,2x,f9.4)'), ' Speedup ratio', time_1/time_2
			WRITE(*,'(a)'),		 ' $$$$$$$$$$$$$$$$$$$$$$$$'

			PRINT*, maxval(abs(sum_1-sum_2))
		END IF
	ELSE
		PRINT*, 'Error! The matrix dimension is not evenly divisible by the number of processors.'
	END IF

END PROGRAM
