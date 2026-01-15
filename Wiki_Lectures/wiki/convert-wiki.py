#!/usr/bin/env python3
"""
Konwertuje pliki wiki-XX.html na wiki-XX.md z pobieraniem obrazków.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import html2text
import sys

WIKI_DIR = "/lab-01/Wiki_Lectures/wiki"

def download_image(url, save_dir, base_url=None, filename=None):
    """Pobiera obrazek i zwraca lokalną ścieżkę."""
    try:
        # Obsłuż różne formaty URL
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            url = 'https://gitlab.com' + url
        elif not url.startswith('http') and base_url:
            # Względny URL - użyj base_url
            url = base_url.rstrip('/') + '/' + url

        # Pobierz oryginalny URL z data-canonical-src jeśli to proxy GitLab
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Wygeneruj nazwę pliku
        if filename is None:
            parsed = urlparse(url)
            filename = os.path.basename(unquote(parsed.path))
            if not filename or filename == '':
                filename = f"image_{hash(url) % 10000}.png"

        # Upewnij się, że ma rozszerzenie
        if '.' not in filename:
            content_type = response.headers.get('content-type', '')
            if 'png' in content_type:
                filename += '.png'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                filename += '.jpg'
            elif 'gif' in content_type:
                filename += '.gif'
            elif 'svg' in content_type:
                filename += '.svg'
            else:
                filename += '.png'

        save_path = os.path.join(save_dir, filename)

        # Zapisz
        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"  Pobrano: {filename}")
        return filename
    except Exception as e:
        print(f"  Błąd pobierania {url}: {e}")
        return None


def convert_wiki_to_md(html_file):
    """Konwertuje plik HTML wiki na Markdown."""
    wiki_num = re.search(r'wiki-(\d+)\.html', html_file)
    if not wiki_num:
        print(f"Nie można określić numeru wiki dla {html_file}")
        return

    num = wiki_num.group(1)
    print(f"\nPrzetwarzanie wiki-{num}...")

    # Base URL dla względnych ścieżek (np. uploads/)
    base_url = f"https://gitlab.com/agh-courses/25/cp/wiki/{num}"

    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Znajdź treść wiki
    content = soup.find('div', class_='js-wiki-content')
    if not content:
        # Spróbuj alternatywnych selektorów
        content = soup.find('div', class_='md')
        if not content:
            content = soup.find('article')
        if not content:
            print(f"  Nie znaleziono treści wiki w {html_file}")
            return

    # Zbierz i pobierz obrazki
    images = content.find_all('img')
    image_map = {}  # stary_url -> nowa_lokalna_ścieżka

    for img in images:
        # Pobierz URL - preferuj data-src (pełna ścieżka GitLab), potem data-canonical-src, potem src
        img_url = img.get('data-src') or img.get('data-canonical-src') or img.get('src')
        if not img_url:
            continue

        # Pobierz obrazek
        local_name = download_image(img_url, WIKI_DIR, base_url=base_url)
        if local_name:
            # Zapisz mapowanie dla wszystkich wariantów URL
            for attr in ['data-canonical-src', 'data-src', 'src']:
                if img.get(attr):
                    image_map[img.get(attr)] = local_name

            # Zaktualizuj src w HTML przed konwersją
            img['src'] = local_name
            # Usuń lazy loading attributes
            if img.get('data-src'):
                del img['data-src']
            if img.get('data-canonical-src'):
                del img['data-canonical-src']
            if img.has_attr('class') and 'lazy' in img.get('class', []):
                img['class'].remove('lazy')

    # Konwertuj na Markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0  # Nie zawijaj linii
    h.unicode_snob = True
    h.skip_internal_links = False
    h.ignore_emphasis = False
    h.mark_code = True

    markdown = h.handle(str(content))

    # Poprawki formatowania
    # Zamień [code]...[/code] na ```...```
    markdown = re.sub(r'\[code\]\s*\n', '```\n', markdown)
    markdown = re.sub(r'\[/code\]', '```', markdown)

    # Usuń nadmiarowe puste linie
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    # Zapisz plik .md
    md_file = os.path.join(WIKI_DIR, f"wiki-{num}.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"  Zapisano: wiki-{num}.md")


def main():
    if len(sys.argv) > 1:
        # Konwertuj konkretny plik
        for arg in sys.argv[1:]:
            if os.path.exists(arg):
                convert_wiki_to_md(arg)
            else:
                # Spróbuj jako numer
                html_file = os.path.join(WIKI_DIR, f"wiki-{arg}.html")
                if os.path.exists(html_file):
                    convert_wiki_to_md(html_file)
                else:
                    print(f"Nie znaleziono: {arg}")
    else:
        # Konwertuj wszystkie pliki wiki-XX.html
        for filename in sorted(os.listdir(WIKI_DIR)):
            if filename.startswith('wiki-') and filename.endswith('.html'):
                convert_wiki_to_md(os.path.join(WIKI_DIR, filename))


if __name__ == '__main__':
    main()