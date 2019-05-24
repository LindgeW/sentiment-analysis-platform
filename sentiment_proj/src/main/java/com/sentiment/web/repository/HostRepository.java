package com.sentiment.web.repository;

import com.sentiment.web.entity.HostName;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

/**
 * Created by WuLinZhi on 2019-04-09.
 */
@Repository
public interface HostRepository extends MongoRepository<HostName, Integer>{
}
