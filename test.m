clearvars
clc

F = [
    1 0.1353 1;
    1 0.3678 0.3678;
    1 0.3678 0.3678;
    1 1 0.1353
];

F_iv = F';

Ans = F_iv*F;

Inversed = inv(Ans);

