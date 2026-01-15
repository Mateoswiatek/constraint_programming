#!/bin/bash

# Skrypt pobierający wiki z GitLab dla każdego laba

cd /mnt/adata-disk/projects/agh/cp

for dir in lab-*; do
    if [[ -d "$dir" ]]; then
        # Wyciągnij numer z nazwy folderu (np. 01 z lab-01)
        num="${dir#lab-}"

        echo "Pobieranie wiki dla $dir (numer: $num)..."

        curl "https://gitlab.com/agh-courses/25/cp/wiki/${num}" \
            --compressed \
            -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0' \
            -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8' \
            -H 'Accept-Language: pl,en-US;q=0.7,en;q=0.3' \
            -H 'Accept-Encoding: gzip, deflate, br, zstd' \
            -H 'Referer: https://gitlab.com/agh-courses/25/cp/wiki' \
            -H 'Connection: keep-alive' \
            -H 'Cookie: _sp_id.6b85=52125196-fb5b-491c-90fb-b370611cde20.1759917764.19.1764709513.1764677783.02c94912-cfdd-469c-a21b-5c210587fc80.9d31f227-3d3d-48f7-9909-d40dbffcce53.d709dd28-c0a7-45d6-91c8-b7f18c3db9a0.1764708273110.160; cf_clearance=rkd0H9zMLhNmIC8iGnE8sOWtysR43UHYE08VYkUctbk-1763547410-1.2.1.1-8zmJ5Wukg0HahhW77SOOM_KfuaKZRGz.ID61PvhhdYI4pBsHgBN4EquVbsMGoBHCuyr20IFOmpx6QXUdnwJwiXJfONrmpDxOdeVeMQ_2z8CYVFZWdUWYChjjFYugoZG2Yf6TtHcsb.r919HPpd101arRNiHNESFY4GalgfP6U.64YqgPhhUe9cQm8lN_Uz1l45hBLzvdRdYaGd..qlAhNFmBt40c9SgUQim9hpk6sio; OptanonConsent=isGpcEnabled=0&datestamp=Wed+Nov+19+2025+11%3A44%3A21+GMT%2B0100+(czas+%C5%9Brodkowoeuropejski+standardowy)&version=202510.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=7f13a723-2c77-48b7-9aa7-6934b8c640c5&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0002%3A1%2CC0004%3A1%2CC0007%3A1%2CC0008%3A1&intType=1&geolocation=PL%3B14&AwaitingReconsent=false; OptanonAlertBoxClosed=2025-10-08T10:05:12.028Z; studio=false; super_sidebar_collapsed=false; gitlab_user=true; gitlab_tier=free; known_sign_in=V3hDTk5vZEI1dENZcDZBYjlZazNWOXJrdmpSU0pJZlpKeHJKLzZDMVFrbzU2ZWtKNUZrRXdZS3VhVEpvejF0UFlSWERKY3VzdCtxWk0waVphaDRCTHZnbi92SzNRb09yd2Uyc1Z3R3FKSEI1VnAwVUFjYXg4eisyR1VPVlZNbGd6RDdrVVp4cWN5Y1M1dWZqd3pkaklnPT0tLVJlVDJPdWR1TmR0aU9sUGh0TUoxQ1E9PQ%3D%3D--27ace99c0a9c96bfa946eb0a3c62f58ac3265668; sidebar_pinned_section_expanded=true; _cfuvid=UrPpp5aNuGDBvrv0fnkhKjoP.ot_UFSYPMJi7251pV4-1764708272480-0.0.1.1-604800000; _gitlab_session=3343ae675cad7725e562ca35c4d9bedb; preferred_language=en; event_filter=all; __cf_bm=nQPtNL.tgcFNQlrYMszzWK5V6SXBg1UUwwQvF5Op2cA-1764709187-1.0.1.1-6S_AZvvAoHiiAxGA27ojJGO3owFFSeGofRCqMFa5r80iansxLTSKEpWOfEedGwuUItJiLTGcBsjAltbgOB0xK1V7GjaWokSYOMzRHMh.RGc; _sp_ses.6b85=*' \
            -H 'Upgrade-Insecure-Requests: 1' \
            -H 'Sec-Fetch-Dest: document' \
            -H 'Sec-Fetch-Mode: navigate' \
            -H 'Sec-Fetch-Site: same-origin' \
            -H 'Sec-Fetch-User: ?1' \
            -H 'Priority: u=0, i' \
            -o "wiki-${num}.html"

        if [[ $? -eq 0 ]]; then
            echo "Zapisano wiki-${num}.html"
        else
            echo "Błąd podczas pobierania wiki-${num}"
        fi

        # Krótka pauza żeby nie obciążać serwera
        sleep 1
    fi
done

echo "Gotowe!"