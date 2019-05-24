package com.sentiment.web;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

//@EnableScheduling  //启动定时任务
@SpringBootApplication
@MapperScan({"com.sentiment.web.repository"})
public class SentimentWebApplication {

	public static void main(String[] args) {
		SpringApplication.run(SentimentWebApplication.class, args);
	}

}
