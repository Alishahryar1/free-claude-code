param(
    [Parameter(Mandatory)][string] $Bundle,
    [switch] $RequireSigned
)

$ErrorActionPreference = 'Stop'
$native = @(Get-ChildItem -LiteralPath $Bundle -Recurse -Force -File | Where-Object {
    $_.Extension -in @('.exe', '.dll', '.pyd', '.ocx', '.sys')
})
if ($native.Count -eq 0) { throw 'The Windows bundle contains no native binaries.' }
foreach ($file in $native) {
    $signature = Get-AuthenticodeSignature -LiteralPath $file.FullName
    if ($signature.Status -eq 'Valid') { continue }
    if ($signature.Status -ne 'NotSigned' -or $RequireSigned) {
        throw "Cannot verify $($file.FullName) ($($signature.Status))."
    }
    $file.FullName
}
