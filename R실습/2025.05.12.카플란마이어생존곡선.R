library(survival)
install.packages("survminer")
library(survminer)

# 카플란 마이어 생존 곡선 1

df <- lung # 폐암 환자 생존률 데이터 
df$event <- ifelse(df$status == 2, 1, 0) # status : 1 censored / 2: 사망
km_fit <- survfit(Surv(time, event) ~1, data = df)
plot(km_fit)
ggsurvplot(km_fit, data=df)

# 카플란 마이어 생존 곡선 2
km_fit_sex <- survfit(Surv(time, event) ~sex, data=df)

ggsurvplot(km_fit_sex, data = df, pval = T, pval.coord = c(500,1),
			risk.table = T, title = "성별에 따른 K-M 생존 곡선")

# 카플란 마이어 생존 곡선 3
df$age_cat <- NA
df$age_cat <- ifelse(df$age < 60, 1, df$age_cat)
df$age_cat <- ifelse(df$age >= 60, 2, df$age_cat)
df$age %>% table
km_fit_age <- survfit(Surv(time, event) ~age_cat, data=df)
ggsurvplot(km_fit_age, data = df, pval = T, pval.coord = c(500, 1),
			risk.table = T, title = "연령대에 따른 K-M 생존 곡선")

# 카플란 마이어 생존 곡선 4
df$age_cat <- NA
df$age_cat <- ifelse(df$age < 70, 1, df$age_cat)
df$age_cat <- ifelse(df$age >= 70, 2, df$age_cat)
df$age %>% table
km_fit_age <- survfit(Surv(time, event) ~age_cat, data=df)
ggsurvplot(km_fit_age, data = df, pval = T, pval.coord = c(500, 1),
			risk.table = T, title = "연령대에 따른 K-M 생존 곡선")

