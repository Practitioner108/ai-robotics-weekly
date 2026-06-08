# 每周一 9:05 自动同步周报到本地并打开浏览器
$repoPath = "C:\Users\ASUS\Documents\资料库\新闻汇总"
$reportDir = "$repoPath\AI与机器人部"
$dateStr = Get-Date -Format "yyyy-MM-dd"

cd $repoPath
git pull origin master | Out-Null

$latestFile = Get-ChildItem -Path $reportDir -Filter "*.html" | Sort-Object Name -Descending | Select-Object -First 1
if ($latestFile) {
    Start-Process $latestFile.FullName
}
