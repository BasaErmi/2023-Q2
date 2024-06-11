### 实验4：DNS配置与应用

#### 【实验目的]

- 理解并掌握DNS工作原理；
- 通过回顾巩固DNS缓存污染攻击和DoS攻击技术；
- 掌握实际中部署和应用DNS方法；
- 通过回顾巩固本地DNS设置限速等基本防御技术；
- 掌握为一个公司或企业注册域名的过程和方法。

#### 【实验环境】

本实验的拓扑分为内网和外网。内网包括2个子网，分别为DMZ区和办公区。相关配置如下：

- 内网 DMZ区:
  Web服务器：IP: 192.168.2.2/24; 网关 192.168.2.1，通过IP地址访问 [http://192.168.2.2](http://192.168.2.2/)
  本地DNS：IP:192.168.2.10
- 内网 办公区:
  教务老师：IP: 192.168.3.102/24; 网关 192.168.3.1
  学生：192.168.3.100/24；网关192.168.3.1
- 外网:
  外网攻击者：IP：192.168.5.100/24
  外网权威DNS: 192.168.4.100

#### 实验内容

域名系统是互联网最重要的基础设施之一。请回顾并巩固我们课堂上讲过的“域名”、“互联网域名空间”、“互联网域名解析”、“DNS协议”、“域名系统常见攻击、攻击原理及防护措施”等知识，以及“DNS缓存污染攻击实验”、“DNS拒绝服务攻击实验”与相应的限速等基本防御技术。本实验主要包括在实际应用的必然碰到的DNS配置技术，以加深对DNS技术、DNS攻击与防御技术的理解。

##### （1）配置内网中的本地域名www.localhost，并在内网中使用。

##### （2）注册可访问的域名www.sysu

#### 【实验步骤】

##### 1. 添加本地域名www.localhost

本实验实现对本地域名的注册和使用。本章前面实验一直是通过IP地址访问网站的，现实中是比较麻烦的。回顾课堂上讲解的DNS知识，域名解析是从递归解析器开始逐级访问权威服务器，以获取域名的解析记录的过程。在这个过程中，权威服务器总是优先检查自己管辖的域文件是否包含被查域名。

为简化操作，本实验将中间的逐个级别的查询过程略去，尝试使用本地DNS作为本地权威，通过添加本地权威解析记录，给拓扑中的web服务器配置本地域名。

##### （1）配置本地DNS

本地DNS机器上安装有域名解析软件bind9。通常情况下bind9的区文件以及配置文件存放在目录 /etc/bind/ 目录下，其中以db开头的为主要的域名存储区文件。bind9的数据加载顺序是named.conf->named.conf.default-zones->db，其中db.local文件为简单示例，它记录域名localhost的区信息。修改db.local文件，添加域名[www.localhost](http://www.localhost/)的A记录，地址为192.168.2.2。



**作答：添加本地域名A记录结果如下：**

<img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606193531822.png" alt="image-20240606193531822" style="zoom:50%;" />



##### (2) 查看并分析 named.conf.default-zones 、named.conf文件

请分析图2、图3中的配置的含义。

<img src="https://www.imool.com.cn/upload/course/c258655a-e829-11ee-86fd-1e8bd935e632/md/7b1fc664-227c-11ef-8649-024267a1371c.jpg" alt="2.jpg" style="zoom:50%;" />

> 图 2 named.conf.default-zones 文件的部分截图

**分析：**

- **zone "localhost"**为一个区域声明，定义了一个DNS区域的开始，区域名为 "localhost" ，用于本地回环地址的解析。
- **type master**指定了该区域的类型为master，表示该服务器是 "localhost" 区域的主服务器，负责管理和提供该区域的dns记录。
- **file "/etc/bind/db.local"**:
  - 指定该区域的数据文件存放在 `/etc/bind/db.local`。



<img src="https://www.imool.com.cn/upload/course/c258655a-e829-11ee-86fd-1e8bd935e632/md/849800da-227c-11ef-a918-024267a1371c.jpg" alt="3.jpg" style="zoom:50%;" />

> 图 3 named.conf 文件的部分截图

**分析：**

使用**include**指令在主配置文件中加载包含其他配置文件，分别加载了 `named.conf.options` 、 `named.conf.local`和 `named.conf.default-zones` 。其中 `named.conf.options` 用于加载全局选项配置，例如递归设置、转发器和DNSSEC选项；`named.conf.local`用于定义自定义区域（zones），例如正向区域和反向区域的配置；`named.conf.local`用于加载默认区域配置。



### （3）令配置生效

配置文件修改完毕后执行命令：

```
#检查配置文件是否正确
named-checkconf
#重启bind9
/etc/init.d/bind9 restart
```

执行命令如下：

![image-20240606201834238](/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606201834238.png)

![image-20240606201909651](/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606201909651.png)



### （4）验证本地注册的域名的有效性

将教务老师以及学生主机的域名解析服务器设置为 192.168.2.10。

教务老师主机是windows系统，可通过更改网络适配器ipv4属性中的域名信息实现。学生主机通过更改 /etc/resolv.conf文件实现。更改完成后，尝试使用nslookup命令或dig命令，在教务老师主机以及学生主机上查询域名[www.localhost](http://www.localhost/)，验证所注册的域名的有效性。记录必要的实验配置修改过程及访问结果。



**教师主机域名解析服务器修改：**

1. **在网络选项的高级网络设置中找到更改适配器选项**：

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606203320329.png" alt="image-20240606203320329" style="zoom:50%;" />

2. **选择网络适配器并打开IPV4属性**：

   - 在适配器界面选择网络（这里只有一个以太网）
   - 然后在网络状态中选择属性
   - 属性中点击IPV4

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606203441549.png" alt="image-20240606203441549" style="zoom:50%;" />

   

3. **在IPV4中设置DNS服务器地址**：

   - 选择“使用下面的DNS服务器地址”。

   - 在“首选DNS服务器”中输入 `192.168.2.10`。

     <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606203652294.png" alt="image-20240606203652294" style="zoom:50%;" />

4. **在命令行使用nslookup查询名字服务器**

   查询结果如下，说明服务器配置成功：

   ![image-20240606213113243](/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606213113243.png)



**学生主机域名解析服务器修改：**

1. **命令行打开`/etc/resolv.conf`修改dns配置**

   注意要用管理员权限

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606204408221.png" alt="image-20240606204408221" style="zoom:50%;" />

2. **更改名字服务器**

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606204545351.png" alt="image-20240606204545351" style="zoom:50%;" />

3. **重启网络服务**

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606211530750.png" alt="image-20240606211530750" style="zoom:50%;" />

4. **使用dig指令查询域名**

   查询结果如下，域名配置成功：

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606213028356.png" alt="image-20240606213028356" style="zoom:50%;" />

## 2. 注册可访问域名www.sysu

参照注册本地域名[www.localhost](http://www.localhost/)的过程，特别是db.local的配置过程，注册可访问域名[www.sysu](http://www.sysu/)，并进行验证。

提示：主要过程包括复制db.local文件，修改localhost为sysu，更改NS记录指向本地DNS，添加[www.sysu](http://www.sysu/)记录指向192.168.2.2，参照localhost将区文件加入named.conf.default-zones，重启bind9。在学生主机或教务老师主机上，可以浏览器访问[www.sysu](http://www.sysu/)进入实验1网站。



1. **首先在本地DNS上将`db.local`复制一份**

   sysu后缀结尾：

   ![image-20240606213323271](/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606213323271.png)

2. **打开复制的文件，然后修改文件内容如下**

   修改localhost为sysu，更改NS记录指向本地DNS，添加[www.sysu](http://www.sysu/)记录指向192.168.2.2：

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606213550745.png" alt="image-20240606213550745" style="zoom:50%;" />

3. **将区文件加入named.conf.default-zones**

   参照localhost将区文件加入named.conf.default-zones，重启bind9：

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606214318782.png" alt="image-20240606214318782" style="zoom: 67%;" />

4. **此时在学生和老师主机上都可以查询到该域名：**

   学生：

   <img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606214405052.png" alt="image-20240606214405052" style="zoom:50%;" />

   老师：

   ![image-20240606214432287](/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606214432287.png)

5. **同时可以通过浏览器访问`www.sysu`进入实验页面**

<img src="/Users/liuguanlin/Library/Application Support/typora-user-images/image-20240606214649340.png" alt="image-20240606214649340" style="zoom:50%;" />