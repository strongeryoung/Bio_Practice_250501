library(survival)
install.packages("survminer")
library(survminer)

cox_model <- coxph(Surv(time, event) ~ sex+age+ph.ecog, data = df)
cox_model %>% summary

cox_model2 <- coxph(Surv(time, event) ~ sex+age+ph.karno, data = df)
cox_model2 %>% summary
cox.zph(cox_model2)