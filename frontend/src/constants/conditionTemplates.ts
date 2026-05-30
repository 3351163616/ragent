export interface ConditionTemplate {
  id: string;
  label: string;
  description: string;
  condition: Record<string, unknown> | null;
}

export interface ConditionFieldReference {
  field: string;
  label: string;
  values?: string;
}

export interface ConditionOperatorReference {
  operator: string;
  label: string;
}

export const CONDITION_TEMPLATES: ConditionTemplate[] = [
  {
    id: "none",
    label: "无条件",
    description: "节点总是执行",
    condition: null
  },
  {
    id: "local-file",
    label: "本地文件",
    description: "仅处理本地上传文件",
    condition: { field: "source_type", op: "eq", value: "file" }
  },
  {
    id: "remote-url",
    label: "URL 内容",
    description: "仅处理 URL 抓取内容",
    condition: { field: "source_type", op: "eq", value: "url" }
  },
  {
    id: "pdf",
    label: "PDF 文档",
    description: "仅处理 PDF MIME 类型",
    condition: { field: "mime_type", op: "contains", value: "pdf" }
  },
  {
    id: "markdown",
    label: "Markdown",
    description: "按文件名后缀匹配",
    condition: { field: "file_name", op: "regex", value: ".*\\.(md|markdown)$" }
  },
  {
    id: "large-file",
    label: "大于 1MB",
    description: "按原始字节大小判断",
    condition: { field: "file_size", op: "gt", value: 1048576 }
  }
];

export const CONDITION_FIELD_REFERENCES: ConditionFieldReference[] = [
  { field: "source_type", label: "文档来源", values: "file, url, feishu, s3" },
  { field: "mime_type", label: "MIME 类型", values: "application/pdf, text/markdown, text/plain" },
  { field: "file_name", label: "文件名", values: "invoice.pdf, handbook.md" },
  { field: "file_size", label: "文件大小", values: "字节数" },
  { field: "source_location", label: "来源地址", values: "本地路径、URL 或对象存储地址" }
];

export const CONDITION_OPERATOR_REFERENCES: ConditionOperatorReference[] = [
  { operator: "eq", label: "等于" },
  { operator: "ne", label: "不等于" },
  { operator: "in", label: "包含于列表" },
  { operator: "contains", label: "包含" },
  { operator: "regex", label: "正则匹配" },
  { operator: "gt/gte", label: "大于 / 大于等于" },
  { operator: "lt/lte", label: "小于 / 小于等于" },
  { operator: "exists", label: "存在" },
  { operator: "not_exists", label: "不存在" }
];
