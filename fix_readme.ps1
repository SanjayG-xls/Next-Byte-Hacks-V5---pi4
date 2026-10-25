$content = Get-Content 'C:\Users\sanjay\Desktop\BytePi4\README.md' -Raw
$lineToRemove = '    <img src="https://cdn.simpleicons.org/arm/0091BD" alt="Arm Logo" height="35" style="margin: 0 15px;"/>'
$content = $content -replace [regex]::Escape($lineToRemove) + "`r`n", ''
Set-Content 'C:\Users\sanjay\Desktop\BytePi4\README.md' -Value $content -NoNewline
