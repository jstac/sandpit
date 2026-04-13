program std_normal
    implicit none
    integer, parameter :: n = 100000000
    real(8), allocatable :: x(:)
    real(8) :: mean, sd, t_start, t_end
    integer :: i

    allocate(x(n))

    call cpu_time(t_start)

    ! Generate standard normals via Box-Muller
    call generate_normals(x, n)

    ! Square them
    do i = 1, n
        x(i) = x(i) ** 2
    end do

    ! Compute standard deviation
    mean = sum(x) / n
    sd = sqrt(sum((x - mean) ** 2) / (n - 1))

    call cpu_time(t_end)

    print '(A, F12.6)', 'Standard deviation: ', sd
    print '(A, F12.6, A)', 'Time: ', t_end - t_start, ' seconds'

    deallocate(x)

contains

    subroutine generate_normals(x, n)
        integer, intent(in) :: n
        real(8), intent(out) :: x(n)
        real(8) :: u1, u2, pi
        integer :: i

        pi = 4.0d0 * atan(1.0d0)
        call random_seed()

        do i = 1, n, 2
            call random_number(u1)
            call random_number(u2)
            x(i) = sqrt(-2.0d0 * log(u1)) * cos(2.0d0 * pi * u2)
            if (i + 1 <= n) then
                x(i+1) = sqrt(-2.0d0 * log(u1)) * sin(2.0d0 * pi * u2)
            end if
        end do
    end subroutine generate_normals

end program std_normal
