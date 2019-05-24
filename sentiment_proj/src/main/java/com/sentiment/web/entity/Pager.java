package com.sentiment.web.entity;

import java.util.List;

/**
 * Created by WuLinZhi on 2019-03-06.
 */
public class Pager {
    private Integer currentPage = 1; //当前页
    private Integer pageSize = 5; //每页显示的记录数
    private Integer totalRows = 0; //总记录数
    private Integer totalPages = 0; //总页数
    private List<Remark> pageData = null; //一页的数据

    public Pager() {
        this.currentPage = 1;
        this.pageSize = 5;
        this.totalRows = 0;
        this.totalPages = 0;
        this.pageData = null;
    }

    public Pager(Integer currentPage, Integer pageSize, Integer totalRows) {
        this.currentPage = currentPage;
        this.pageSize = pageSize;
        this.totalRows = totalRows;
        if(totalRows % pageSize == 0){
            this.totalPages = totalRows / pageSize;
        }else{
            this.totalPages = totalRows / pageSize + 1;
        }
        this.pageData = null;
    }

    public void setTotalRows(Integer totalRows) {
        this.totalRows = totalRows;
    }

    public Integer getTotalRows() {
        return this.totalRows;
    }

    public Integer getTotalPages() {
        if(totalRows % pageSize == 0){
            this.totalPages = totalRows / pageSize;
        }else{
            this.totalPages = totalRows / pageSize + 1;
        }
        return this.totalPages;
    }

    public Integer getPageSize() {
        return pageSize;
    }

    public void setPageSize(Integer pageSize) {
        this.pageSize = pageSize;
    }

    public Integer getCurrentPage() {
        return currentPage;
    }

    public void setCurrentPage(Integer currentPage) {
        if (currentPage < 1){
            this.currentPage = 1;
        }else if(currentPage > getTotalPages()){
            this.currentPage = getTotalPages();
        }else {
            this.currentPage = currentPage;
        }
    }

    public void setPageData(List<Remark> dataset) {
        int firstIndex = (currentPage-1) * pageSize;
        int lastIndex = currentPage * pageSize;
        if (lastIndex > getTotalRows()){
            lastIndex = getTotalRows();
        }
        this.pageData = dataset.subList(firstIndex, lastIndex);
    }

    public List<Remark> getPageData() {
        return this.pageData;
    }
}
