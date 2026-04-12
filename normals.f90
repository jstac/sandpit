program normals
    implicit none

    integer, parameter :: n = 100000000
    integer, parameter :: blk = 1000000
    real(8), dimension(blk) :: u1, u2
    real(8) :: z_sq, sum1, sum2, mean, std_dev, pi
    integer :: i, j, count_start, count_end, count_rate

    pi = 4.0d0 * atan(1.0d0)

    call system_clock(count_start, count_rate)

    sum1 = 0.0d0
    sum2 = 0.0d0

    do i = 1, n / blk
        call random_number(u1)
        call random_number(u2)
        do j = 1, blk
            z_sq = (-2.0d0 * log(u1(j))) * cos(2.0d0 * pi * u2(j))**2
            sum1 = sum1 + z_sq
            sum2 = sum2 + z_sq * z_sq
        end do
    end do

    mean = sum1 / n
    std_dev = sqrt(sum2 / n - mean * mean)

    call system_clock(count_end)

    print '(A, F12.6)', 'Mean of squared normals:   ', mean
    print '(A, F12.6)', 'Std dev of squared normals:', std_dev
    print '(A, F10.4, A)', 'Time: ', &
        real(count_end - count_start) / real(count_rate), ' seconds'

end program normals
