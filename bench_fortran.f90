program bench_normal
    implicit none
    integer, parameter :: n = 100000000
    real(8), allocatable :: u1(:), u2(:), z(:), z_sq(:)
    real(8) :: mean, std, t_start, t_end
    real(8), parameter :: pi = 3.141592653589793d0
    integer :: i

    allocate(u1(n), u2(n), z(n), z_sq(n))

    call cpu_time(t_start)

    ! Generate uniform random numbers
    call random_number(u1)
    call random_number(u2)

    ! Box-Muller transform to get standard normals
    do i = 1, n
        z(i) = sqrt(-2.0d0 * log(u1(i))) * cos(2.0d0 * pi * u2(i))
    end do

    ! Square them
    do i = 1, n
        z_sq(i) = z(i) * z(i)
    end do

    ! Compute mean
    mean = sum(z_sq) / n

    ! Compute standard deviation
    std = sqrt(sum((z_sq - mean)**2) / n)

    call cpu_time(t_end)

    print '(A, F12.6)', 'Standard deviation: ', std
    print '(A, F12.6, A)', 'Time:               ', t_end - t_start, ' seconds'

    deallocate(u1, u2, z, z_sq)

end program bench_normal
