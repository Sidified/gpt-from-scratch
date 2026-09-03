# What each ablated model learned

- uniform guessing: 4.1744 nats
- unigram distribution: 3.3091 nats

| run | val loss | vs unigram | distinct-4gram |
|---|---|---|---|
| baseline | 1.5571 | -1.752 | 0.931 |
| no_pos_emb | 1.8466 | -1.4625 | 0.956 |
| no_residual | 3.3592 | +0.0501 | 0.964 |
| no_layernorm | 1.7929 | -1.5161 | 0.899 |


### baseline  (val 1.5571)

```


ANGELO:
And low against yet that seek of thee.
Stirrah, what hath barnity foul crown.
Was not how he will weak, my lord, I mean with to thy heart and more
As their less hand; and may sometime to the friends wails
All and tears the firecive o' the pi
```

### no_pos_emb  (val 1.8466)

```

When thy bridlerial, is not the diself,
Fear what hat thy dagain
My would he should to beh dinghom.

First Cizen:
Your m hen heavens, tof in he me milindine, ath ireles, hein fat strid or te, anend hin anondermises w.
Hon me hitenser onchir speak; an
```

### no_residual  (val 3.3592)

```

snrnt ih tri  ori eh    naerter aireo  oaeo  etarthrta s teataeaohs ar  thhe  h hrt tetrar ethrhoaoe  r  r   a eansaa o  oeo he ostserto  ia h iot   t noitit aes ireeesen tie tat sntiaror ts  eoen    neahoi era sesei ri h ee oetehshr on  ih   tarr ni
```

### no_layernorm  (val 1.7929)

```

What thy bridleriaght son, be madiest but of than hath streatin of the haphe
Have the that dinght, for where he fast, will mus
How shall firsh both stand to ways is the what that strivior the do him him:
Will and welp in you his hus would by stay;
Or
```
