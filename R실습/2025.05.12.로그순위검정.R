library(survival)

run_data <- data.frame(
	group = c("A", "A", "A", "A", "B", "B", "B", "B"),
	time = c(15.2, 17.5, 14.0, 20.0, 20.0, 18.0, 17.0, 19.5), # (추적 시간)
	event = c(1,1,1,0,0,1,1,1)) # 사건 발생 여부 (1 = 발생, 0 = 검열) 

log_rank_test <- survdiff(Surv(time, event) ~ group, 
							data = run_data)

log_rank_test