package com.sentiment.web.repository;

import com.sentiment.web.entity.Remark;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Created by WuLinZhi on 2019-03-08.
 */
/*
先继承MongoRepository<T, TD>接口，其中T为仓库保存的bean类，TD为该bean的唯一标识的类型，一般为ObjectId。之后在service中注入该接口就可以使用，无需实现里面的方法，spring会根据定义的规则自动生成。
但是MongoRepository实现了的只是最基本的增删改查的功能，要想增加额外的查询方法，可以按照以下规则定义接口的方法。
自定义查询方法，格式为“findBy+字段名+方法后缀”，方法传进的参数即字段的值，此外还支持分页查询，通过传进一个Pageable对象，返回Page集合。
 */
@Repository
public interface RemarkRepository extends MongoRepository<Remark, String> { //bean对象自动映射到数据库中的document
    //根据评论内容进行模糊查询
    List<Remark> findByContentLike(String content);

    //评论内容不包含
    List<Remark> findByContentNot(String content);

    //分页查询全部
//    Page<Remark> findAll(Pageable pageable);

    //根据评论内容分页查找
    Page<Remark> findByContentLike(String content, Pageable pageable);
}
