Age = transpose([data.haemods.age]);
HR = transpose([data.haemods.HR]);
SV = transpose([data.haemods.SV]);
SBP_a = transpose([data.haemods.SBP_a]);
DBP_a = transpose([data.haemods.DBP_a]);
PWV_bf = transpose([data.haemods.PWV_bf]);

age_hr_sbp_dbp = cat(2, Age, HR, SBP_a, DBP_a);
age_hr_sv_sbp_dbp = cat(2, Age, HR, SV, SBP_a, DBP_a);

writematrix(age_hr_sbp_dbp, 'age_hr_sbp_dbp.csv');
writematrix(age_hr_sv_sbp_dbp, 'age_hr_sv_sbp_dbp.csv');
writematrix(PWV_bf, 'pwv_bf.csv');