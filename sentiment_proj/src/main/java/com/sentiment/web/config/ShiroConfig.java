package com.sentiment.web.config;

import org.apache.shiro.mgt.SecurityManager;
import org.apache.shiro.spring.LifecycleBeanPostProcessor;
import org.apache.shiro.spring.security.interceptor.AuthorizationAttributeSourceAdvisor;
import org.apache.shiro.spring.web.ShiroFilterFactoryBean;
import org.apache.shiro.web.mgt.DefaultWebSecurityManager;
import org.springframework.aop.framework.autoproxy.DefaultAdvisorAutoProxyCreator;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.DependsOn;

import java.util.LinkedHashMap;
import java.util.Map;
/*
Shiro的三大核心组件：
　　1、Subject------------------当前用户
　　2、SecurityManage--------管理所有的Subject
　　3、Reamls------------------权限信息的验证
    我们需要实现Reamls的Authentication与Authorization，其中Authentication用于验证用户身份，Authorization用于授权访问控制。
　　Shiro核心是通过Filter来实现，就好像SpringMVC用DispatchServlet来做控制一样。
    既然使用Filter，那么我们也可以猜到，Shiro是通过URL规则来进行过滤和权限校验，所以我们需要定义一系列的URL规则和访问权限。
　　另外通过Shiro提供的会话管理可以获取Session中的信息，Shiro也提供缓存功能，使用CacheManage来管理。

    集成Shiro可分为如下几个步骤：
    　　1、pom.xml中添加Shiro依赖
        2、注入ShiroFilterFactory和SecurityManage
    　　3、身份认证
    　　4、权限控制　　
    anon:所有url都都可以匿名访问;
　　authc: 需要认证才能进行访问;
　　user:配置记住我或认证通过可以访问；
 */

/**
 * Created by WuLinZhi on 2019-03-18.
 */

@Configuration
public class ShiroConfig {
    @Bean
    public ShiroFilterFactoryBean shiroFilter(SecurityManager securityManager){
        System.out.println("ShiroConfig.shiroFilter()");
        //ShiroFilterFactoryBean 处理拦截资源文件问题
        ShiroFilterFactoryBean shiroFilterFactoryBean=new ShiroFilterFactoryBean();
        //设置安全管理器
        shiroFilterFactoryBean.setSecurityManager(securityManager);
        //默认跳转到登陆页面
        shiroFilterFactoryBean.setLoginUrl("/login");
        //登陆成功后的页面
        shiroFilterFactoryBean.setSuccessUrl("/index");
        //未授权页面 403
        shiroFilterFactoryBean.setUnauthorizedUrl("/403");
        //自定义过滤器
//        Map<String, Filter> filterMap=new LinkedHashMap<>();
//        shiroFilterFactoryBean.setFilters(filterMap);
        //权限控制map
        Map<String,String> filterChainDefinitionMap=new LinkedHashMap<>();
        // 配置不会被拦截的链接 顺序判断
        filterChainDefinitionMap.put("/favicon.ico", "anon"); //防止登录成功之后下载favicon.ico
        filterChainDefinitionMap.put("/", "anon");
        filterChainDefinitionMap.put("/index", "anon");
        filterChainDefinitionMap.put("/login", "anon");
        //自定义logout需要注释掉该行
//        filterChainDefinitionMap.put("/logout", "logout"); //配置退出过滤器,其中的具体的退出代码Shiro已经替我们实现了

        filterChainDefinitionMap.put("/static/**", "anon");
        filterChainDefinitionMap.put("/css/**", "anon");
        filterChainDefinitionMap.put("/img/**", "anon");
        filterChainDefinitionMap.put("/js/**", "anon");

        //普通用户、游客
        filterChainDefinitionMap.put("/user/**", "anon");
        //管理员，需要角色权限 “admin”
        filterChainDefinitionMap.put("/admin/**", "roles[admin]");
        //<!-- 过滤链定义，从上向下顺序执行，一般将/**放在最为下边 -->:这是一个坑呢，一不小心代码就不好使了;
//        //<!-- authc:所有url都必须认证通过才可以访问; anon:所有url都都可以匿名访问-->
        filterChainDefinitionMap.put("/**", "authc");
        shiroFilterFactoryBean.setFilterChainDefinitionMap(filterChainDefinitionMap);
        return shiroFilterFactoryBean;
    }

    @Bean
    public SecurityManager securityManager(){
        DefaultWebSecurityManager securityManager =  new DefaultWebSecurityManager();
        securityManager.setRealm(myShiroRealm());
        return securityManager;
    }

    @Bean
    public UserRealm myShiroRealm(){
        return new UserRealm();
    }

    //开启Shiro的注解(如@RequiresRoles,@RequiresPermissions)
    @Bean
    public DefaultAdvisorAutoProxyCreator defaultAdvisorAutoProxyCreator(){
        DefaultAdvisorAutoProxyCreator app=new DefaultAdvisorAutoProxyCreator();
        app.setProxyTargetClass(true);
        return app;
    }

    //开启aop注解支持
    @Bean
    public AuthorizationAttributeSourceAdvisor authorizationAttributeSourceAdvisor(SecurityManager securityManager) {
        AuthorizationAttributeSourceAdvisor authorizationAttributeSourceAdvisor = new AuthorizationAttributeSourceAdvisor();
        authorizationAttributeSourceAdvisor.setSecurityManager(securityManager);
        return authorizationAttributeSourceAdvisor;
    }

    /**
     * Shiro生命周期处理器
     * @return
     */
    @Bean
    public LifecycleBeanPostProcessor lifecycleBeanPostProcessor(){
        return new LifecycleBeanPostProcessor();
    }

    /**
     * 自动创建代理
     * @return
     */
    @Bean
    @DependsOn({"lifecycleBeanPostProcessor"})
    public DefaultAdvisorAutoProxyCreator advisorAutoProxyCreator(){
        DefaultAdvisorAutoProxyCreator advisorAutoProxyCreator = new DefaultAdvisorAutoProxyCreator();
        advisorAutoProxyCreator.setProxyTargetClass(true);
        return advisorAutoProxyCreator;
    }
}

