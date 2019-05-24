package com.sentiment.web.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.Map;

/**
 * 实现对Python服务端的HTTP请求
 */
public class HttpRequestUtil {
	/**
	 * 向指定URL发送GET方法的请求
	 * 
	 * @param url
	 *            发送请求的URL
	 * @return URL所代表远程资源的响应
	 */

	public static String sendGet(String url) {
		String result = "";
		BufferedReader in = null;
		try {
			System.out.println(url);
			HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection(); // 打开和URL之间的连接
//			conn.setRequestProperty("Charsert", "UTF-8"); //设置请求编码
			conn.setUseCaches(false); 
			// 发送GET请求必须设置如下两行
            conn.setDoInput(true);
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(6000);
			// 建立实际的连接
			conn.connect();
			// 获取所有响应头字段
			 Map< String, List< String >> map = conn.getHeaderFields();
			// 遍历所有的响应头字段
			 for (String key: map.keySet()) {
				 System.out.println(key + "--->" + map.get(key));
			 }
			// 定义BufferedReader输入流来读取URL的响应
			in = new BufferedReader(new InputStreamReader(conn.getInputStream(), "gbk"));
			String line;
			while ((line = in.readLine()) != null) {
				result += line;
			}
		} catch (Exception e) {
			System.out.println("发送GET请求出现异常！" + e);
			e.printStackTrace();
		}
		// 使用finally块来关闭输入流
		finally {
			try {
				if (in != null) {
					in.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return result;
	}

	/**
	 * 向指定URL发送POST方法的请求
	 * 
	 * @param url
	 *            发送请求的URL
	 * @param content
	 *            请求参数，请求参数应该是name1=value1&name2=value2的形式。
	 * @return URL所代表远程资源的响应
	 */
	public static String sendPost(String url, Object content, String charsetName) {
		PrintWriter out = null;
		BufferedReader in = null;
		HttpURLConnection conn = null;
		String result = "";
		try {
			// 打开和URL之间的连接
			conn = (HttpURLConnection) new URL(url).openConnection();			
			// 设置通用的请求属性
			conn.setRequestMethod("POST");
			//禁用缓存
			conn.setUseCaches(false); 
			//容许输出
			conn.setDoOutput(true);
			//容许输入
			conn.setDoInput(true);
			/*
			 * 协议规定 POST 提交的数据必须放在消息主体（entity-body）中，
			 * 但协议并没有规定数据必须使用什么编码方式。
			 * 但是，数据发送出去还要服务端解析成功才有意义。
			 * 服务端通常是根据请求头（headers）中的 Content-Type 字段来获知请求中的消息主体是用何种方式编码，再对主体进行解析。
			 * POST提交数据方案，包含了 Content-Type 和消息主体编码方式两部分。
			 * 
			 * Content-Type 被指定为 application/x-www-form-urlencoded；
			 * 其次，提交的数据按照 key1=val1&key2=val2 的方式进行编码，key和 val都进行了URL转码。
			 * 大部分服务端语言都对这种方式有很好的支持。
			 */
			//application/x-www-form-urlencoded   multipart/form-data  application/json
//			conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded;charset=utf-8");
			conn.setRequestProperty("Content-Type", "application/json;charset="+charsetName);
			conn.setRequestProperty("accept","application/json");
			conn.setRequestProperty("Charset", charsetName); //设置请求编码
			conn.setRequestProperty("Connection", "Keep-Alive");
			conn.connect();
			// 获取URLConnection对象对应的输出流
			out = new PrintWriter(conn.getOutputStream());
			// 发送请求参数(正文内容其实跟get的URL中 '?'后的参数字符串一致)
//			out.write(content);
			out.print(content);
			// flush输出流的缓冲
			out.flush();
			// 定义BufferedReader输入流来读取URL的响应
			in = new BufferedReader(new InputStreamReader(conn.getInputStream(), charsetName));
			String line;
			while ((line = in.readLine()) != null) {
				result += line;
			}
		} catch (Exception e) {
			System.out.println("发送POST请求出现异常！" + e);
			e.printStackTrace();
		}
		// 使用finally块来关闭输出流、输入流
		finally {
			try {
				if (out != null) {
					out.close();
				}
				if (in != null) {
					in.close();
				}
				if (conn != null){
					conn.disconnect();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return result;
	}

//	public static void main(String[] args) {
//		Map<String, Object> dto = new HashMap<String, Object>();
//        dto.put("name", "张三丰");
//        dto.put("age", "10");
//        Object param = JSONObject.toJSON(dto);
//		System.out.println(param.toString());
//
////		String param = "name=张三丰&age=10";
//
//
//		String getResult = HttpRequestUtil.sendPost("http://127.0.0.1:5000/todo", param);
//		System.out.println(JSONObject.parseObject(getResult));
//	}
}
