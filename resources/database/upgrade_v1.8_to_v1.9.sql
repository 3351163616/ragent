INSERT INTO t_intent_node (
    id, kb_id, intent_code, name, level, parent_code, description, examples, collection_name, top_k, kind, sort_order, enabled, deleted
)
SELECT 'demo-biz-security',
       '2053404940851757056',
       'demo-biz-security',
       '业务数据安全',
       1,
       'demo-biz',
       '业务系统数据安全治理，包括访问控制、数据分级、导出水印、加密、审计合规、传输接口安全和安全事件响应。',
       '["业务系统数据安全怎么做？","L3/L4 数据导出要哪些管控？","高敏附件下载和水印追踪有什么要求？","账号权限生命周期如何管理？"]',
       'ragentdemobiz',
       6,
       0,
       23,
       1,
       0
WHERE NOT EXISTS (
    SELECT 1 FROM t_intent_node WHERE intent_code = 'demo-biz-security' AND deleted = 0
);

UPDATE t_intent_node
SET parent_code = 'demo-group-hr',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code IN (
      'hr-company-rules',
      'hr-leave-attendance',
      'hr-probation-performance',
      'hr-promotion-transfer',
      'hr-recruitment',
      'hr-salary-benefits',
      'hr-training'
  );

UPDATE t_intent_node
SET parent_code = 'demo-group-it',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code IN (
      'it-account',
      'it-device',
      'it-email',
      'it-escalation',
      'it-meeting',
      'it-network-wifi',
      'it-permission',
      'it-printer-scanner',
      'it-vpn'
  );

UPDATE t_intent_node
SET parent_code = 'demo-group-finance',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code IN ('finance-invoice');

UPDATE t_intent_node
SET parent_code = 'demo-biz-security',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code IN (
      'access-control',
      'audit-compliance',
      'data-classification',
      'data-encryption',
      'data-security-overview',
      'export-watermark',
      'incident-response',
      'model-algorithm-security',
      'transmission-security'
  );

UPDATE t_intent_node
SET description = '企业邮箱收发异常、共享邮箱、客户端配置、垃圾邮件、钓鱼邮件识别，以及误点邮件链接后的安全上报、断网、改密和服务台处置。',
    examples = '["收不到外部邮件怎么办","共享邮箱怎么设置","收到疑似钓鱼邮件怎么处理","误点邮件链接后要不要断网改密","垃圾邮件过滤规则怎么配"]',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code = 'it-email';

UPDATE t_intent_node
SET description = 'Windows 或 macOS 添加办公网络打印机，系统设置/打印机与扫描仪入口，AirPrint 或厂商驱动选择，打印队列、扫描到邮箱和安全打印排查。',
    examples = '["macOS 上怎么添加办公网络打印机","添加打印机驱动优先选 AirPrint 吗","打印队列卡住怎么办","怎么把扫描文件发到邮箱","安全打印怎么取件"]',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code = 'it-printer-scanner';

UPDATE t_intent_node
SET description = '业务系统权限申请和访问授权，包括 VPN 或网络已连通但系统提示无权限、账号角色授权、外包或临时访问、批量开通、审批与权限回收。',
    examples = '["VPN 已连接但系统提示无权限怎么办","访问 OA 提示没有权限怎么申请","外包人员临时访问系统怎么开通","项目权限需要批量开通","账号角色权限到期怎么回收"]',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code = 'it-permission';

UPDATE t_intent_node
SET description = 'VPN 客户端安装、连接超时、握手失败、无法建立隧道、内网网络连通性和合规使用排查；不处理已连通后的业务系统授权问题。',
    examples = '["VPN 连不上怎么办","VPN 连接超时或握手失败怎么排查","在家怎么连公司网络","VPN 已连接但内网地址打不开怎么办","VPN 客户端安装失败"]',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code = 'it-vpn';

UPDATE t_intent_node
SET description = '社会招聘、校园招聘、Offer、背景调查、入职办理，以及新员工入职后第 1 周、2 周内、满 1 个月的 HR 跟进、五险一金、试用期目标和回访。',
    examples = '["社招流程是什么","候选人多久能发 Offer","入职需要带什么材料","新员工入职第 1 周 HR 要跟进什么","入职 2 周内如何制定试用期目标","入职满 1 个月 HR 是否要回访"]',
    update_time = CURRENT_TIMESTAMP
WHERE deleted = 0
  AND intent_code = 'hr-recruitment';
