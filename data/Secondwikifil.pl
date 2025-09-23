#!/usr/bin/perl

$/ = ">";                     # input record separator
while (<>) {
  if (/<text /) { $text = 1; }    # keep only between <text> ... </text>
  if (/#redirect/i) { $text = 0; }  # skip redirects
  if ($text) {

    # Stop at end of text
    if (/<\/text>/) { $text = 0; }

    # --- Rimozioni / sostituzioni ---
    s/<.*?>//g;             # remove xml tags -> replace with space
    s/&amp;/&/g;             # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*//g;     # remove url, keep visible text later
    s/\|thumb//ig;
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/$1/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|//g;   # remove wiki url, preserve visible text
    s/\{\{[^\}]*\}\}//g;   # remove {{icons}} and {tables}
    s/\{[^\}]*\}//g;
    s/\[|\]//g;            # remove [ and ]
    s/&[^;]*;//g;          # remove URL encoded chars
    s/<div[^>]*>.*?<\/div>//sg;
    s/<[^>]*>//g;
    s/<references\s*\/>//ig;

    s/^==[^=]+==\s*//g;
    s/'''[^']*'''//g;
    s/''//g;                   # elimina doppi apici

    # --- Simboli speciali ---
    s/%/ percent /g;
    s/\$/ dollar /g;
    s/\&/ and /g;

    # --- Tenere lettere, numeri, ., ?, !, -, spazi ---
    s/[^a-zA-Z0-9\-'.?! ]//g;

    # --- Pulizia spazi ---
    s/\s+/ /g;   # collapse multiple spaces
    s/^\s+|\s+$//g; # trim

        # rimuove intestazioni tipo ==See also==
    s/^==.*?==\s*$//mg;

    # rimuove sottosezioni ===Historical events===
    s/^===.*?===\s*$//mg;

    # rimuove liste puntate * ...
    s/^\*.*$//mg;

    # rimuove liste numerate # ...
    s/^#.*$//mg;

    # rimuove template {{...}}
    s/\{\{[^\}]*\}\}//g;

    # rimuove categorie [[Category:...]]
    s/\[\[Category:[^\]]*\]\]//ig;
    s/\[\[[a-z\-]+:[^\]]*\]\]//ig;

    # rimuove link interlingua [[it:...]], [[en:...]], ecc.
    s/\[\[[a-z\-]+:[^\]]*\]\]//ig;

    # elimina righe vuote
    s/^\s*$//mg;

    # Output
    print $_, "\n";
  }
}

