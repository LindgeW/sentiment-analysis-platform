package com.sentiment.web.controller;

import com.sentiment.web.entity.Remark;
import com.sentiment.web.entity.RespEntity;
import com.sentiment.web.entity.RespStatus;
import com.sentiment.web.service.RemarkService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletResponse;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Created by WuLinZhi on 2019-03-09.
 */
@Controller
@RequestMapping("/admin")
public class RemarkController {
    private static final String prefix = "admin";
    @Autowired
    private RemarkService remarkService;

    @GetMapping("/bg")
    public String toBackground() {
//        System.out.println((String) SecurityUtils.getSubject().getPrincipal());
        return prefix + "/bg";
    }

    @GetMapping("/review")
    public String review() {
        return prefix + "/review";
    }

    @PostMapping("/monitor")
    @ResponseBody
    public Map<String, Integer> remarkConunt(){
        List<Remark> remarks = remarkService.findAll();
        return remarkService.magicCount(remarks);
    }

    @PostMapping("/get_remarks")
    @ResponseBody
    public List<Remark> getRemarks(){
        return remarkService.findAll();
    }

    @PostMapping("/del_one")
    @ResponseBody
    public RespEntity delOne(@RequestBody Remark rm){
        remarkService.deleteOne(rm);
        return new RespEntity(RespStatus.SUCCESS);
    }

    @PostMapping("/del_remarks")
    @ResponseBody
    public RespEntity delRemarks(@RequestBody List<Remark> rms){
        remarkService.deleteAll(rms);
        return new RespEntity(RespStatus.SUCCESS);
    }

    @PostMapping("/update_one")
    @ResponseBody
    public RespEntity updateRemark(@RequestBody Remark remark){
        //若新增数据的主键已经存在，则会对当前已经存在的数据进行修改操作
        remarkService.save(remark);
        return new RespEntity(RespStatus.SUCCESS);
    }

    @PostMapping("/update_remarks")
    @ResponseBody
    public RespEntity updateRemarks(@RequestBody List<Remark> remarks){
        //若新增数据的主键已经存在，则会对当前已经存在的数据进行修改操作
        remarkService.saveAll(remarks);
        return new RespEntity(RespStatus.SUCCESS);
    }

    @GetMapping("/export_from_db")
    public void export(HttpServletResponse response) throws IOException {
        List<Remark> remarkList = getRemarks();
        System.out.println("downloading.....");
        String defaultFileName = "untitled.csv";
        response.setContentType(MediaType.APPLICATION_OCTET_STREAM_VALUE);
        response.setHeader("Content-Disposition", "attachment; filename=" + defaultFileName);

        BufferedOutputStream bufOs = new BufferedOutputStream(response.getOutputStream());
        for(Remark rm : remarkList){
            bufOs.write(rm.toString().getBytes());
            bufOs.flush();
        }
        bufOs.close();
    }
}
