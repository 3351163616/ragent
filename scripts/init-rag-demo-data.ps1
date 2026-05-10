param(
    [switch]$Help,
    [string]$BaseUrl = "http://localhost:9090/api/ragent",
    [string]$Username = "admin",
    [string]$Password = "admin",
    [string]$EmbeddingModel = "bge-m3",
    [string]$GroupKnowledgeBaseName = "RAG Demo - 集团信息化",
    [string]$BizKnowledgeBaseName = "RAG Demo - 业务系统",
    [string]$GroupCollectionName = "ragentdemogroup",
    [string]$BizCollectionName = "ragentdemobiz",
    [switch]$ForceUpload,
    [switch]$SkipChunk,
    [switch]$WaitChunk
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Net.Http

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")
$KnowledgeDir = Join-Path $RootDir "resources\docs\knowledge"
$ApiBase = $BaseUrl.TrimEnd("/")

function Show-Usage {
    Write-Host "Ragent RAG demo 数据初始化脚本"
    Write-Host ""
    Write-Host "用法："
    Write-Host "  .\scripts\init-rag-demo-data.cmd"
    Write-Host "  .\scripts\init-rag-demo-data.cmd -WaitChunk"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\init-rag-demo-data.ps1 -BaseUrl http://localhost:9090/api/ragent"
    Write-Host ""
    Write-Host "常用参数："
    Write-Host "  -BaseUrl <url>          后端 API 地址，默认 http://localhost:9090/api/ragent"
    Write-Host "  -Username <name>        登录用户名，默认 admin"
    Write-Host "  -Password <password>    登录密码，默认 admin"
    Write-Host "  -EmbeddingModel <id>    知识库 embedding 模型，默认 bge-m3"
    Write-Host "  -ForceUpload            删除同名旧文档后重新上传"
    Write-Host "  -SkipChunk              只建知识库/意图/映射/文档，不触发分块向量化"
    Write-Host "  -WaitChunk              触发分块后等待异步任务完成"
    Write-Host "  -Help                   显示本帮助"
}

function Write-Step {
    param([string]$Message)
    Write-Host "[init] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[ ok ] $Message" -ForegroundColor Green
}

function Write-WarnLine {
    param([string]$Message)
    Write-Host "[warn] $Message" -ForegroundColor Yellow
}

function Read-ResultData {
    param(
        [Parameter(Mandatory = $true)]
        $Response,
        [string]$Operation = "API 请求"
    )

    if ($null -eq $Response) {
        throw "$Operation 未返回响应"
    }
    if ($Response.PSObject.Properties.Name -contains "code" -and $Response.code -ne "0") {
        $message = if ($Response.message) { $Response.message } else { "未知错误" }
        throw "$Operation 失败：$message"
    }
    if ($Response.PSObject.Properties.Name -contains "data") {
        return $Response.data
    }
    return $Response
}

function Invoke-RagentJson {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("GET", "POST", "PUT", "DELETE", "PATCH")]
        [string]$Method,
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [object]$Body,
        [hashtable]$Headers = @{},
        [string]$Operation = "API 请求"
    )

    $uri = "$ApiBase$Path"
    $requestHeaders = @{}
    foreach ($key in $Headers.Keys) {
        $requestHeaders[$key] = $Headers[$key]
    }

    $params = @{
        Method = $Method
        Uri = $uri
        Headers = $requestHeaders
    }
    if ($PSBoundParameters.ContainsKey("Body")) {
        $params.ContentType = "application/json; charset=utf-8"
        $params.Body = $Body | ConvertTo-Json -Depth 20
    }

    try {
        $response = Invoke-RestMethod @params
        return Read-ResultData -Response $response -Operation $Operation
    } catch {
        throw "$Operation 失败：$($_.Exception.Message)"
    }
}

function Invoke-RagentMultipart {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [hashtable]$Fields,
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$File,
        [hashtable]$Headers = @{},
        [string]$Operation = "上传文件"
    )

    $client = [System.Net.Http.HttpClient]::new()
    $content = [System.Net.Http.MultipartFormDataContent]::new()
    $fileStream = $null

    try {
        foreach ($key in $Headers.Keys) {
            $client.DefaultRequestHeaders.Remove($key) | Out-Null
            $client.DefaultRequestHeaders.Add($key, [string]$Headers[$key])
        }
        foreach ($key in $Fields.Keys) {
            $fieldContent = [System.Net.Http.StringContent]::new([string]$Fields[$key], [System.Text.Encoding]::UTF8)
            $content.Add($fieldContent, $key)
        }

        $fileStream = [System.IO.File]::OpenRead($File.FullName)
        $fileContent = [System.Net.Http.StreamContent]::new($fileStream)
        $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("text/markdown")
        $content.Add($fileContent, "file", $File.Name)

        $responseMessage = $client.PostAsync("$ApiBase$Path", $content).GetAwaiter().GetResult()
        $responseText = $responseMessage.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not $responseMessage.IsSuccessStatusCode) {
            throw "HTTP $([int]$responseMessage.StatusCode) $($responseMessage.ReasonPhrase)：$responseText"
        }
        $response = $responseText | ConvertFrom-Json
        return Read-ResultData -Response $response -Operation $Operation
    } catch {
        throw "$Operation 失败：$($_.Exception.Message)"
    } finally {
        if ($fileStream) {
            $fileStream.Dispose()
        }
        $content.Dispose()
        $client.Dispose()
    }
}

function Get-PageRecords {
    param($PageData)
    if ($null -eq $PageData) {
        return @()
    }
    if ($PageData.records) {
        return @($PageData.records)
    }
    return @()
}

function Get-TreeNodeByCode {
    param(
        [array]$Nodes,
        [string]$IntentCode
    )

    foreach ($node in $Nodes) {
        if ($node.intentCode -eq $IntentCode) {
            return $node
        }
        if ($node.children) {
            $matched = Get-TreeNodeByCode -Nodes @($node.children) -IntentCode $IntentCode
            if ($null -ne $matched) {
                return $matched
            }
        }
    }
    return $null
}

function Ensure-KnowledgeBase {
    param(
        [string]$Name,
        [string]$CollectionName
    )

    $page = Invoke-RagentJson -Method GET -Path "/knowledge-base?current=1&size=200" -Headers $AuthHeaders -Operation "查询知识库列表"
    $existing = Get-PageRecords $page | Where-Object { $_.name -eq $Name -or $_.collectionName -eq $CollectionName } | Select-Object -First 1
    if ($existing) {
        Write-Ok "复用知识库：$($existing.name) ($($existing.id), $($existing.collectionName))"
        return $existing
    }

    $id = Invoke-RagentJson -Method POST -Path "/knowledge-base" -Headers $AuthHeaders -Operation "创建知识库 $Name" -Body @{
        name = $Name
        embeddingModel = $EmbeddingModel
        collectionName = $CollectionName
    }
    $created = Invoke-RagentJson -Method GET -Path "/knowledge-base/$id" -Headers $AuthHeaders -Operation "查询新建知识库 $Name"
    Write-Ok "创建知识库：$($created.name) ($($created.id), $($created.collectionName))"
    return $created
}

function Ensure-IntentNode {
    param(
        [hashtable]$Node
    )

    $tree = Invoke-RagentJson -Method GET -Path "/intent-tree/trees" -Headers $AuthHeaders -Operation "查询意图树"
    $existing = Get-TreeNodeByCode -Nodes @($tree) -IntentCode $Node.intentCode
    if ($existing) {
        $update = @{}
        foreach ($key in @(
            "name", "level", "parentCode", "description", "examples", "kbId", "topK", "kind",
            "sortOrder", "enabled", "promptSnippet", "promptTemplate", "paramPromptTemplate"
        )) {
            if ($Node.ContainsKey($key)) {
                $update[$key] = $Node[$key]
            }
        }
        Invoke-RagentJson -Method PUT -Path "/intent-tree/$($existing.id)" -Headers $AuthHeaders -Operation "更新意图 $($Node.intentCode)" -Body $update | Out-Null
        Write-Ok "更新意图：$($Node.intentCode)"
        return
    }

    Invoke-RagentJson -Method POST -Path "/intent-tree" -Headers $AuthHeaders -Operation "创建意图 $($Node.intentCode)" -Body $Node | Out-Null
    Write-Ok "创建意图：$($Node.intentCode)"
}

function Ensure-Mapping {
    param(
        [string]$SourceTerm,
        [string]$TargetTerm,
        [int]$Priority = 100,
        [string]$Remark = "RAG demo 初始化"
    )

    $keyword = [uri]::EscapeDataString($SourceTerm)
    $page = Invoke-RagentJson -Method GET -Path "/mappings?current=1&size=50&keyword=$keyword" -Headers $AuthHeaders -Operation "查询术语映射 $SourceTerm"
    $existing = Get-PageRecords $page | Where-Object { $_.sourceTerm -eq $SourceTerm } | Select-Object -First 1
    $body = @{
        sourceTerm = $SourceTerm
        targetTerm = $TargetTerm
        matchType = 1
        priority = $Priority
        enabled = $true
        remark = $Remark
    }

    if ($existing) {
        Invoke-RagentJson -Method PUT -Path "/mappings/$($existing.id)" -Headers $AuthHeaders -Operation "更新术语映射 $SourceTerm" -Body $body | Out-Null
        Write-Ok "更新术语映射：$SourceTerm -> $TargetTerm"
        return
    }

    Invoke-RagentJson -Method POST -Path "/mappings" -Headers $AuthHeaders -Operation "创建术语映射 $SourceTerm" -Body $body | Out-Null
    Write-Ok "创建术语映射：$SourceTerm -> $TargetTerm"
}

function Upload-DocumentIfNeeded {
    param(
        [string]$KnowledgeBaseId,
        [string]$FilePath
    )

    $file = Get-Item -LiteralPath $FilePath
    $encodedKeyword = [uri]::EscapeDataString($file.Name)
    $page = Invoke-RagentJson -Method GET -Path "/knowledge-base/$KnowledgeBaseId/docs?current=1&size=20&keyword=$encodedKeyword" -Headers $AuthHeaders -Operation "查询文档 $($file.Name)"
    $existing = Get-PageRecords $page | Where-Object { $_.docName -eq $file.Name -and $_.deleted -ne 1 } | Select-Object -First 1

    if ($existing -and -not $ForceUpload) {
        Write-Ok "跳过已存在文档：$($file.Name) ($($existing.id), status=$($existing.status), chunks=$($existing.chunkCount))"
        return $existing
    }

    if ($existing -and $ForceUpload) {
        Invoke-RagentJson -Method DELETE -Path "/knowledge-base/docs/$($existing.id)" -Headers $AuthHeaders -Operation "删除旧文档 $($file.Name)" | Out-Null
        Write-Ok "已删除旧文档：$($file.Name)"
    }

    $fields = @{
        sourceType = "file"
        processMode = "chunk"
        chunkStrategy = "structure_aware"
        chunkConfig = '{"targetChars":1400,"overlapChars":0,"maxChars":1800,"minChars":600}'
    }
    $doc = Invoke-RagentMultipart -Path "/knowledge-base/$KnowledgeBaseId/docs/upload" -Headers $AuthHeaders -Fields $fields -File $file -Operation "上传文档 $($file.Name)"
    Write-Ok "上传文档：$($file.Name) ($($doc.id))"
    return $doc
}

function Start-DocumentChunk {
    param($Document)

    if ($SkipChunk) {
        return
    }
    if ($Document.status -eq "success" -and $Document.chunkCount -gt 0 -and -not $ForceUpload) {
        Write-Ok "跳过已分块文档：$($Document.docName)"
        return
    }
    if ($Document.status -eq "running") {
        Write-WarnLine "文档正在分块中：$($Document.docName)"
        return
    }

    Invoke-RagentJson -Method POST -Path "/knowledge-base/docs/$($Document.id)/chunk" -Headers $AuthHeaders -Operation "触发文档分块 $($Document.docName)" | Out-Null
    Write-Ok "已触发分块：$($Document.docName)"
}

function Wait-DocumentsChunked {
    param(
        [array]$Documents,
        [int]$TimeoutSeconds = 300
    )

    if (-not $WaitChunk -or $SkipChunk) {
        return
    }

    Write-Step "等待文档分块完成，最长 $TimeoutSeconds 秒"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $states = @()
        foreach ($doc in $Documents) {
            $latest = Invoke-RagentJson -Method GET -Path "/knowledge-base/docs/$($doc.id)" -Headers $AuthHeaders -Operation "查询文档状态 $($doc.docName)"
            $states += $latest
        }

        $running = @($states | Where-Object { $_.status -eq "running" -or $_.status -eq "pending" })
        $failed = @($states | Where-Object { $_.status -eq "failed" })
        if ($failed.Count -gt 0) {
            $names = ($failed | ForEach-Object { $_.docName }) -join ", "
            throw "以下文档分块失败：$names"
        }
        if ($running.Count -eq 0) {
            Write-Ok "文档分块已完成"
            return
        }

        $summary = ($states | ForEach-Object { "$($_.docName):$($_.status)/$($_.chunkCount)" }) -join "; "
        Write-Host "       $summary"
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)

    Write-WarnLine "等待超时，分块任务可能仍在后台执行。可稍后到知识库文档页面查看状态。"
}

if ($Help) {
    Show-Usage
    exit 0
}

if (-not (Test-Path -LiteralPath $KnowledgeDir)) {
    throw "示例知识库目录不存在：$KnowledgeDir"
}

foreach ($collectionName in @($GroupCollectionName, $BizCollectionName)) {
    if ($collectionName -notmatch "^[a-z0-9]+$") {
        throw "Collection 名称只能包含小写英文字母和数字：$collectionName"
    }
}

Write-Step "登录 $ApiBase"
$loginData = Invoke-RagentJson -Method POST -Path "/auth/login" -Operation "登录" -Body @{
    username = $Username
    password = $Password
}
$AuthHeaders = @{ Authorization = $loginData.token }
Write-Ok "登录成功：$Username"

Write-Step "创建或复用 demo 知识库"
$groupKb = Ensure-KnowledgeBase -Name $GroupKnowledgeBaseName -CollectionName $GroupCollectionName
$bizKb = Ensure-KnowledgeBase -Name $BizKnowledgeBaseName -CollectionName $BizCollectionName

Write-Step "初始化意图树"
$systemPrompt = "你是 Ragent，一个支持知识库问答的 Agentic RAG 助手。用户询问系统能力时，请直接、简洁、准确地说明：你可以基于已上传/已入库的知识库文档进行检索问答；知识库的创建、文档上传、入库、分块、向量化、检索等能力由系统前后端提供。不要编造当前用户已经上传了哪些具体文档；如果用户想继续操作，引导其到知识库/文档管理入口上传或导入资料。"

$intentNodes = @(
    @{ intentCode = "demo-group"; name = "集团信息化"; level = 0; kind = 0; sortOrder = 10; enabled = 1; description = "企业内部制度、IT支持、人事、财务等集团信息化知识"; kbId = $groupKb.id },
    @{ intentCode = "demo-group-hr"; name = "人事制度"; level = 1; parentCode = "demo-group"; kind = 0; sortOrder = 11; enabled = 1; description = "招聘、培训、规章制度、薪资福利、请假考勤等人事相关问题"; examples = @("请假流程是什么？", "加班到凌晨第二天几点上班？", "薪资福利有哪些？"); kbId = $groupKb.id; topK = 6 },
    @{ intentCode = "demo-group-it"; name = "IT支持"; level = 1; parentCode = "demo-group"; kind = 0; sortOrder = 12; enabled = 1; description = "账号密码、VPN、打印机、邮箱、网络、办公软件等 IT 支持问题"; examples = @("VPN 连不上怎么办？", "电脑账号密码忘了怎么处理？", "打印机怎么连接？"); kbId = $groupKb.id; topK = 6 },
    @{ intentCode = "demo-group-finance"; name = "财务发票"; level = 1; parentCode = "demo-group"; kind = 0; sortOrder = 13; enabled = 1; description = "开票信息、发票抬头、税号、银行账号等财务开票问题"; examples = @("发票抬头是什么？", "公司税号是多少？"); kbId = $groupKb.id; topK = 6 },
    @{ intentCode = "demo-biz"; name = "业务系统"; level = 0; kind = 0; sortOrder = 20; enabled = 1; description = "OA、保险等业务系统的功能、架构和数据安全规范"; kbId = $bizKb.id },
    @{ intentCode = "demo-biz-oa"; name = "OA系统"; level = 1; parentCode = "demo-biz"; kind = 0; sortOrder = 21; enabled = 1; description = "OA系统功能、安全治理、访问控制、审计、数据分类分级等问题"; examples = @("OA 系统如何做权限控制？", "OA 数据安全目标是什么？"); kbId = $bizKb.id; topK = 6 },
    @{ intentCode = "demo-biz-ins"; name = "保险系统"; level = 1; parentCode = "demo-biz"; kind = 0; sortOrder = 22; enabled = 1; description = "互联网保险系统的数据安全、投保核保理赔数据保护、访问控制与合规规范"; examples = @("保险系统如何保护敏感信息？", "理赔数据如何做权限控制？"); kbId = $bizKb.id; topK = 6 },
    @{ intentCode = "sys"; name = "系统交互"; level = 0; kind = 1; sortOrder = 100; enabled = 1; description = "问候、助手能力介绍、系统操作引导等非知识库问答" },
    @{ intentCode = "sys-welcome"; name = "欢迎与问候"; level = 1; parentCode = "sys"; kind = 1; sortOrder = 101; enabled = 1; description = "用户打招呼、问候、寒暄"; examples = @("你好", "hello", "早上好", "在吗", "嗨") },
    @{ intentCode = "sys-about-bot"; name = "关于助手"; level = 1; parentCode = "sys"; kind = 1; sortOrder = 102; enabled = 1; description = "询问助手是谁、能做什么、有什么能力"; examples = @("你是谁？", "你是做什么的？", "你能帮我做什么？") },
    @{ intentCode = "sys-capability"; name = "知识库能力说明"; level = 1; parentCode = "sys"; kind = 1; sortOrder = 103; enabled = 1; description = "用户询问是否支持知识库、文档上传、入库、分块、向量化、检索问答等系统能力"; examples = @("你可以存知识库吗？", "能上传文档吗？", "怎么导入知识库？", "你支持哪些知识库功能？"); promptSnippet = $systemPrompt; promptTemplate = $systemPrompt }
)

foreach ($node in $intentNodes) {
    Ensure-IntentNode -Node $node
}

Write-Step "初始化术语映射"
$mappings = @(
    @{ source = "OA"; target = "OA系统"; priority = 10 },
    @{ source = "oa"; target = "OA系统"; priority = 10 },
    @{ source = "VPN"; target = "VPN"; priority = 20 },
    @{ source = "vpn"; target = "VPN"; priority = 20 },
    @{ source = "报销"; target = "财务报销"; priority = 30 },
    @{ source = "发票"; target = "开票信息"; priority = 30 },
    @{ source = "保险"; target = "互联网保险系统"; priority = 40 },
    @{ source = "知识库"; target = "知识库"; priority = 50 }
)
foreach ($mapping in $mappings) {
    Ensure-Mapping -SourceTerm $mapping.source -TargetTerm $mapping.target -Priority $mapping.priority
}

Write-Step "上传示例文档"
$groupDocs = @(
    "group\hr\公司规章制度.md",
    "group\hr\人事制度.md",
    "group\hr\薪资与福利政策.md",
    "group\hr\招聘信息.md",
    "group\hr\员工培训.md",
    "group\it\IT支持.md",
    "group\group-finance\开票信息.md"
)
$bizDocs = @(
    "biz\biz-oa\OA系统数据安全规范文档.md",
    "biz\biz-ins\互联网保险系统数据安全规范.md"
)

$documents = @()
foreach ($relativePath in $groupDocs) {
    $documents += Upload-DocumentIfNeeded -KnowledgeBaseId $groupKb.id -FilePath (Join-Path $KnowledgeDir $relativePath)
}
foreach ($relativePath in $bizDocs) {
    $documents += Upload-DocumentIfNeeded -KnowledgeBaseId $bizKb.id -FilePath (Join-Path $KnowledgeDir $relativePath)
}

Write-Step "触发文档分块与向量化"
foreach ($doc in $documents) {
    Start-DocumentChunk -Document $doc
}
Wait-DocumentsChunked -Documents $documents

Write-Host ""
Write-Ok "RAG demo 数据初始化完成"
Write-Host "       知识库：$($groupKb.name) / $($bizKb.name)"
Write-Host "       示例问题：VPN 连不上怎么办？"
Write-Host "       示例问题：OA 系统如何控制不同角色的权限？"
Write-Host "       示例问题：你可以存知识库吗？"
if (-not $WaitChunk -and -not $SkipChunk) {
    Write-WarnLine "分块与向量化是异步任务，刚跑完脚本时文档可能还在 running。需要同步等待可加 -WaitChunk。"
}
