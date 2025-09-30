#!/usr/bin/perl

$/ = ">";                               # sets the input record separator to ">"
while (<>) {
  if (/<text /) { $text = 1; }          # start processing inside <text>
  if (/#redirect/i) { $text = 0; }      # skip redirects
  if ($text) {

    # Stop at end of text
    if (/<\/text>/) { $text = 0; }      # stop at the end of <text>


    s/<.*?>//g;                         # remove all XML tags
    s/&amp;/&/g;                        # decode HTML special characters
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;              # remove <ref> ... </ref> references
    s/<[^>]*>//g;                       # Remove any remaining tags
    s/<div[^>]*>.*?<\/div>//sg;         # Remove <div> blocks
    s/<references\s*\/>//ig;            # Remove self-closing <references/>

    s/\[http:[^] ]*//g;                             # Remove external links but keep visible text
    s/\|thumb//ig;
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/$1/ig;        # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;                   # remove links to other languages
    s/\[\[[^\|\]]*\|//g;   # Remove templates like {{...}}
    s/\{\{[^\}]*\}\}//g;   # remove {{icons}} and {tables}
    s/\{[^\}]*\}//g;
    s/\[|\]//g;            # remove [ and ]
    s/&[^;]*;//g;          # remove URL encoded chars

    s/<[^>]*>//g;
    s/<references\s*\/>//ig;    # Remove self-closing <references/>

    s/^==[^=]+==\s*//g;
    s/'''[^']*'''//g;
    s/''//g;                   # remove double apostrophes

    # --- Special symbols ---
    s/%/ percent /g;
    s/\$/ dollar /g;
    s/\&/ and /g;

    # --- Keep letters, numbers, ., ?, !, -, spaces ---
    s/[^a-zA-Z0-9\-'.?! ]//g;

    # --- Clean up spaces ---
    s/\s+/ /g;   # collapse multiple spaces
    s/^\s+|\s+$//g; # trim

    # remove headers like ==See also==
    s/^==.*?==\s*$//mg;

    # remove subsections ===Historical events===
    s/^===.*?===\s*$//mg;

    # remove bullet lists * ...
    s/^\*.*$//mg;

    # remove numbered lists # ...
    s/^#.*$//mg;

    # remove templates {{...}}
    s/\{\{[^\}]*\}\}//g;

    # remove categories [[Category:...]]
    s/\[\[Category:[^\]]*\]\]//ig;
    s/\[\[[a-z\-]+:[^\]]*\]\]//ig;

    # remove language links [[it:...]], [[en:...]], etc.
    s/\[\[[a-z\-]+:[^\]]*\]\]//ig;

    # remove empty lines
    s/^\s*$//mg;

    # Remove words that start with double dashes (--h, --N, --Z, etc.)
    s/\b--[A-Za-z0-9]+\b//g;

    # Remove standalone double dashes
    s/\s--\s/ /g;

    # Remove isolated single capital letters followed by a dot (U., D., B.)
    s/\b[A-Z]\.\b//g;

    # Output
    print $_, "\n";
  }
}
