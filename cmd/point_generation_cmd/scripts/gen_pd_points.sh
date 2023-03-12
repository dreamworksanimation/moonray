#foreach x ( `seq 1 100000` )
foreach x ( `seq 300000 400000` )
    ./pd_generation -seed $x -out pd_points/p$x.dat
end
