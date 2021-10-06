begin
    w3m -o ssl_verify_server=false https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/1-10000
    w3m -o ssl_verify_server=false https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/10001-20000
    w3m -o ssl_verify_server=false https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/20001-30000
    w3m -o ssl_verify_server=false https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/30001-40000
end |awk 'NF == 3 && $2 != "-" && $1 == v + 1 {print; v = $1}'
