program std_normals
    implicit none

    integer, parameter :: n = 100000000
    real(8), allocatable :: x(:)
    real(8) :: mean, variance, std_dev
    real(8) :: u1, u2, pi
    integer :: i
    integer :: t_start, t_end, count_rate

    pi = 4.0d0 * atan(1.0d0)

    allocate(x(n))

    call system_clock(t_start, count_rate)

    ! Generate standard normals via Box-Muller transform
    do i = 1, n, 2
        call random_number(u1)
        call random_number(u2)
        x(i)   = sqrt(-2.0d0 * log(u1)) * cos(2.0d0 * pi * u2)
        if (i + 1 <= n) then
            x(i+1) = sqrt(-2.0d0 * log(u1)) * sin(2.0d0 * pi * u2)
        end if
    end do

    ! Square them
    x = x * x

    ! Compute mean
    mean = sum(x) / dble(n)

    ! Compute standard deviation
    variance = sum((x - mean)**2) / dble(n - 1)
    std_dev = sqrt(variance)

    call system_clock(t_end)

    print '(A, F12.6)', 'Mean of squared normals:   ', mean
    print '(A, F12.6)', 'Std dev of squared normals:', std_dev
    print '(A, F10.4, A)', 'Elapsed time: ', &
        dble(t_end - t_start) / dble(count_rate), ' seconds'

    deallocate(x)

end program std_normals
