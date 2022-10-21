#!/usr/bin/env python3
import json
import re
import os.path
import pickle
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from operator import itemgetter
from copy import copy

from sudachipy import tokenizer, dictionary
import jaconv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties

from adjustText import adjust_text

from emoji import UNICODE_EMOJI

matplotlib.use("module://mplcairo.macosx")

TIMELINE = os.path.join(os.path.dirname(__file__), "timeline.pickle")
TIMEZONE = timezone(timedelta(hours=9), "JST")

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro", "Yu Gothic", "Meirio", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]
emoji_prop = FontProperties(fname="/System/Library/Fonts/Apple Color Emoji.ttc")

UNICODE_EMOJI = UNICODE_EMOJI["en"]

# (ward to plot, line style, color)
RTA_EMOTES = (
    ("rtaClap", "-", "#ec7087"),
    ("rtaPray", "-", "#f7f97a"),
    (("rtaGl", "GL"), "-", "#5cc200"),
    (("rtaGg", "GG"), "-", "#ff381c"),
    ("rtaCheer", "-", "#ffbe00"),
    ("rtaHatena", "-", "#ffb5a1"),
    ("rtaR", "-", "white"),
    (("rtaCry", "BibleThump"), "-", "#5ec6ff"),

    ("rtaListen", "-.", "#5eb0ff"),
    ("rtaKabe", "-.", "#bf927a"),
    ("rtaFear", "-.", "#8aa0ec"),
    (("rtaRedbull", "rtaRedbull2"), "-.", "#98b0df"),
    ("rtaPokan", "-.", "#838187"),
    ("rtaGogo", "-.", "#df4f69"),
    # ("rtaBanana", ":", "#f3f905"),
    # ("rtaBatsu", ":", "#5aafdd"),
    # ("rtaShogi", ":", "#c68d46"),
    # ("rtaIizo", ":", "#0f9619"),

    ("rtaHello", "-.", "#ff3291"),
    ("rtaHmm", "-.", "#fcc7b9"),
    ("rtaPog", "-.", "#f8c900"),
    ("rtaMaru", ":", "#c80730"),
    ("rtaFire", ":", "#E56124"),
    ("rtaIce", ":", "#CAEEFA"),
    ("rtaThunder", ":", "#F5D219"),
    ("rtaPoison", ":", "#9F65B2"),
    ("rtaWind", ":", "#C4F897"),
    # ("rtaOko", "-.", "#d20025"),
    ("rtaWut", ":", "#d97f8d"),
    ("rtaPolice", ":", "#7891b8"),
    # ("rtaChan", "-.", "green"),
    # ("rtaKappa", "-.", "#ffeae2"),

    # ("rtaSleep", "-.", "#ff8000"),
    # ("rtaCafe", "--", "#a44242"),
    # ("rtaDot", "--", "#ff3291"),

    # ("rtaShi", ":", "#8aa0ec"),
    # ("rtaGift", ":", "white"),
    # ("rtaAnkimo", ":", "#f92218"),

    ("rtaFrameperfect", "--", "#ff7401"),
    ("rtaPixelperfect", "--", "#ffa300"),
    (("草", "ｗｗｗ", "LUL"), "--", "#1e9100"),
    ("無敵時間", "--", "red"),
    # ("かわいい", "--", "#ff3291"),
    (("セーヌ", "ナイスセーヌ"), "--", "#7A4520"),
    # ("EDF", "--", "gray"),
    (("PokSuicune", "スイクン"), "--", "#c38cdc"),
    ("Squid4", "--", "#80d2b4"),
    # ("石油王", "--", "yellow"),
    # (("ｆｆｆ", "稲"), "--", "#ffeab4"),
    # ("〜ケンカ", "--", "orange"),
    # (("Kappu", "カップ", "日清食品"), "--", "#f9bc71"),
    # ("サクラチル", "--", "#ffe0e0"),

)
VOCABULARY = set(w for w, _, _, in RTA_EMOTES if isinstance(w, str))
VOCABULARY |= set(chain(*(w for w, _, _, in RTA_EMOTES if isinstance(w, tuple))))

# (title, movie start time as timestamp, offset hour, min, sec)
GAMES = (
    ("始まりの\nあいさつ", 1660155165.7, 0, 15, 50, 'right'),
    ("地球防衛軍5", 1660155165.7, 0, 17, 50),
    ("AKIBA’S TRIP2", 1660155165.7, 1, 47, 44),
    ("Castlevania\n白夜の協奏曲", 1660155165.7, 2, 46, 57),
    ("Ori and the Blind Forest: Definitive Edition", 1660155165.7, 3, 18, 34),
    ("VVVVVV", 1660155165.7, 4, 15, 52),
    ("SKYGUNNER", 1660155165.7, 4, 54, 5, 'right'),
    ("Minecraft", 1660155165.7, 5, 47, 49),
    ("ウッチャンナンチャンの炎のチャレンジャー\n電流イライラ棒", 1660155165.7, 7, 10, 1),
    ("ゼルダの伝説", 1660155165.7, 8, 6, 6),
    ("ゼルダの伝説 ムジュラの仮面", 1660155165.7, 9, 26, 56),
    ("ゼルダの伝説 トワイライトプリンセス", 1660155165.7, 10, 52, 33),
    ("まじかるキッズどろぴー", 1660155165.7, 14, 8, 46, 'right'),
    ("ロックマン7 宿命の対決！", 1660155165.7, 14, 38, 35),
    ("ゾンビ式 英語力蘇生術 ENGLISH OF THE DEAD", 1660155165.7, 15, 47, 30),
    ("大乱闘スマッシュブラザーズDX", 1660155165.7, 17, 2, 53),
    ("カービィファイターズZ", 1660155165.7, 17, 50, 32),
    ("星のカービィディスカバリー", 1660155165.7, 19, 11, 20),
    ("ポケモン不思議のダンジョン 青の救助隊", 1660155165.7, 21, 27, 17),
    ("ポケモンレンジャー", 1660155165.7, 24, 15, 2),
    ("スーパーマリオブラザーズ", 1660155165.7, 27, 32, 5),
    ("スーパーマリオサンシャイン", 1660155165.7, 28, 16, 29),
    ("ヨッシーのロードハンティング", 1660155165.7, 31, 46, 14),
    ("ロックマンX", 1660155165.7, 32, 44, 38),
    ("忍者龍剣伝3作リレー", 1660155165.7, 33, 37, 27),
    ("テイルズオブシンフォニア", 1660155165.7, 34, 38, 18),

    ("マリオカート ダブルダッシュ!!", 1660308239.7, 0, 7, 45),
    ("ファイナルファンタジーVII インターナショナル for PC", 1660308239.7, 1, 2, 2),
    ("聖剣伝説Legend of Mana\nHDリマスター", 1660308239.7, 3, 16, 11),
    ("エストポリス伝記Ⅱ", 1660308239.7, 5, 46, 55),
    ("アンシャントロマン 〜Power of Dark Side〜", 1660308239.7, 9, 27, 10),
    ("Miner Ultra Adventures\nDirector’s Cut", 1660308239.7, 12, 19, 55),
    ("ごく普通の鹿のゲーム\nDEEEER Simulator", 1660308239.7, 12, 53, 20),
    ("たけしの挑戦状", 1660308239.7, 13, 20, 17),
    ("Paris Chase", 1660308239.7, 13, 55, 18),
    ("ソニック\nライダーズ\nシューティング\nスターストーリー", 1660308239.7, 14, 27, 25),
    ("ドルアーガの塔", 1660308239.7, 14, 50, 1),
    ("Chinatris", 1660308239.7, 15, 31, 28),
    ("EQUALINE", 1660308239.7, 16, 3, 47),
    ("ぷよぷよ～ん", 1660308239.7, 16, 48, 59),
    ("ストリートファイター2\nターボ\nハイパー ファイティング", 1660308239.7, 17, 28, 27),
    ("最後の忍道", 1660308239.7, 17, 59, 37),
    ("ENDER LILIES: Quietus of the Knights", 1660308239.7, 18, 28, 40),
    ("弾銃フィーバロン", 1660308239.7, 20, 1, 3),
    ("ポケモンスタジアム金銀", 1660308239.7, 20, 38, 14),
    ("Elden Ring", 1660308239.7, 21, 33, 38),
    ("Cato", 1660308239.7, 23, 3, 54),

    ("ファイアーエムブレム紋章の謎", 1660392593.7, 0, 7, 28),
    ("マリオ＆ルイージRPG", 1660392593.7, 3, 15, 4),
    ("スーパーマリオブラザーズ３", 1660392593.7, 4, 56, 43),
    ("Yooka-Laylee", 1660392593.7, 6, 22, 2, 'right'),
    ("ツヴァイ!!", 1660392593.7, 6, 59, 16),
    ("機動戦士ガンダム\nギレンの野望 アクシズの脅威V", 1660392593.7, 8, 53, 53),
    ("溶鉄のマルフーシャ", 1660392593.7, 9, 51, 17),
    ("バトルトード&ダブルドラゴン", 1660392593.7, 11, 7, 27, 'right'),
    ("GeoGuessr", 1660392593.7, 11, 48, 40),
    ("AREA 4643", 1660392593.7, 12, 27, 54),
    ("Ghostwire: Tokyo", 1660392593.7, 13, 9, 12),
    ("ザ・ハウス・オブ・ザ・デッド3", 1660392593.7, 14, 45, 21),
    ("Little\nNightmares", 1660392593.7, 15, 30, 27),
    ("スーパーマリオオデッセイ", 1660392593.7, 16, 23, 26, 'right'),
    ("シャリーのアトリエ～黄昏の海の錬金術士～DX", 1660392593.7, 17, 53, 52),
    ("ノスタルジア（コナステ版）", 1660392593.7, 19, 54, 48),
    ("ソニックカラーズアルティメット", 1660392593.7, 20, 49, 47),
    ("鬼武者２", 1660392593.7, 22, 8, 54),
    ("白き鋼鉄のX(イクス)2", 1660392593.7, 23, 28, 15),
    ("クレヨンしんちゃん『オラと博士の夏休み』\n～おわらない七日間の旅～", 1660392593.7, 24, 13, 37),
    ("Good Job!", 1660392593.7, 26, 21, 17),
    ("ビッグトーナメントゴルフ", 1660392593.7, 27, 50, 22),
    ("魂斗羅スピリッツ", 1660392593.7, 29, 3, 12, 'right'),
    ("がんばれゴエモン ゆき姫救出絵巻", 1660392593.7, 29, 21, 4),
    ("アトランチスの謎", 1660392593.7, 30, 18, 40),
    ("Bomb Chicken", 1660392593.7, 30, 40, 48),
    ("爆ボンバーマンシリーズリレー", 1660392593.7, 32, 3, 37, 'right'),
    ("ドラゴンクエストIII そして伝説へ…", 1660392593.7, 33, 26, 52),
    ("終わりのあいさつ", 1660392593.7, 37, 12, 16, "right")
)


class Game:
    def __init__(self, name, t, h, m, s, align="left"):
        self.name = name
        self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s)
        self.align = align


GAMES = tuple(Game(*args) for args in GAMES)

WINDOWSIZE = 1
WINDOW = timedelta(seconds=WINDOWSIZE)
AVR_WINDOW = 60
PER_SECONDS = 60
FIND_WINDOW = 15
DOMINATION_RATE = 0.6
COUNT_THRESHOLD = 37.5

DPI = 200
ROW = 5
PAGES = 4
YMAX = 700
WIDTH = 3840
HEIGHT = 2160


FONT_COLOR = "white"
FRAME_COLOR = "white"
BACKGROUND_COLOR = "#3f6392"
FACE_COLOR = "#274064"
ARROW_COLOR = "#ffff79"
MESSAGE_FILL_COLOR = "#1e0d0b"


class Message:
    _tokenizer = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C

    pns = (
        "無敵時間",
        "石油王",
        "任意の不動産",
        "いつもの",
        "ナイスセーヌ",
        "ファイナルステージ",
        "ヨシ！",
        "やきう",
        "なっとるやろがい",
        "やったか",
        "地獄の業火でな",
        "慣性の法則",
        "興味ないね",
        "ファイナルハンマー",
        "ご照覧",
        "猫は液体",
        "メガトン構文",
        "ファイナルバトル",
        "バイト穴",
        "おばあちゃん",
        "RTAマシーン"
    )
    pn_patterns = (
        (re.compile("[\u30A1-\u30FF]+ケンカ"), "〜ケンカ"),
        (re.compile("^素晴らしい.*今日が来.*$"), "素晴らしい今日が来た（ｒｙ"),
        (re.compile("^ふあふあ[ ､　、]パウパウ.*$"), "ふあふあ､パウパウ（ｒｙ")
    )
    stop_words = (
        "Squid2",
        "する"
    )

    @classmethod
    def _tokenize(cls, text):
        return cls._tokenizer.tokenize(text, cls._mode)

    def __init__(self, raw):
        self.name = raw["author"]["name"]
        if "emotes" in raw:
            self.emotes = set(e["name"] for e in raw["emotes"]
                              if e["name"] not in self.stop_words)
        else:
            self.emotes = set()
        self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000).replace(tzinfo=TIMEZONE)

        self.message = raw["message"]
        self.msg = set()

        message = self.message
        for emote in self.emotes:
            message = message.replace(emote, "")
        for stop in self.stop_words:
            message = message.replace(stop, "")

        #
        for pattern, replace in self.pn_patterns:
            match = pattern.findall(message)
            if match:
                self.msg.add(replace)
                if pattern.pattern.startswith('^') and pattern.pattern.endswith('$'):
                    message = ''
                else:
                    for m in match:
                        message.replace(m, "")

        #
        for pn in self.pns:
            if pn in message:
                self.msg.add(pn)
                message = message.replace(pn, "")

        #
        message = jaconv.h2z(message)

        # (名詞 or 動詞) (+助動詞)を取り出す
        parts = []
        currentpart = None
        for m in self._tokenize(message):
            part = m.part_of_speech()[0]

            if currentpart:
                if part == "助動詞":
                    parts.append(m.surface())
                else:
                    self.msg.add(''.join(parts))
                    parts = []
                    if part in ("名詞", "動詞"):
                        currentpart = part
                        parts.append(m.surface())
                    else:
                        currentpart = None
            else:
                if part in ("名詞", "動詞"):
                    currentpart = part
                    parts.append(m.surface())

        if parts:
            self.msg.add(''.join(parts))

        #
        kusa = False
        for word in copy(self.msg):
            if set(word) & set(('w', 'ｗ')):
                kusa = True
                self.msg.remove(word)
        if kusa:
            self.msg.add("ｗｗｗ")

        message = message.strip()
        if not self.msg and message:
            self.msg.add(message)

    def __len__(self):
        return len(self.msg)

    @property
    def words(self):
        return self.msg | self.emotes


def _parse_chat(paths):
    messages = []
    for p in paths:
        with open(p) as f, Pool() as pool:
            j = json.load(f)
            messages += list(pool.map(Message, j, len(j) // pool._processes))

    timeline = []
    currentwindow = messages[0].datetime.replace(microsecond=0) + WINDOW
    _messages = []
    for m in messages:
        if m.datetime <= currentwindow:
            _messages.append(m)
        else:
            timeline.append((currentwindow, *_make_timepoint(_messages)))
            while True:
                currentwindow += WINDOW
                if m.datetime <= currentwindow:
                    _messages = [m]
                    break
                else:
                    timeline.append((currentwindow, 0, Counter()))

    if _messages:
        timeline.append((currentwindow, *_make_timepoint(_messages)))

    return timeline


def _make_timepoint(messages):
    total = len(messages)
    counts = Counter(_ for _ in chain(*(m.words for m in messages)))

    return total, counts


def _load_timeline(paths):
    if os.path.exists(TIMELINE):
        with open(TIMELINE, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = _parse_chat(paths)
        with open(TIMELINE, "wb") as f:
            pickle.dump(timeline, f)

    return timeline


def _save_counts(timeline):
    _, _, counters = zip(*timeline)

    counter = Counter()
    for c in counters:
        counter.update(c)

    with open("words.tab", 'w') as f:
        for w, c in sorted(counter.items(), key=itemgetter(1), reverse=True):
            print(w, c, sep='\t', file=f)


def _plot(timeline):
    for npage in range(1, 1 + PAGES):
        chunklen = int(len(timeline) / PAGES / ROW)

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR
        plt.subplots_adjust(left=0.07, bottom=0.05, top=0.92)

        for i in range(1, 1 + ROW):
            nrow = i + ROW * (npage - 1)
            f, t = chunklen * (nrow - 1), chunklen * nrow
            x, c, y = zip(*timeline[f:t])
            _x = tuple(t.replace(tzinfo=None) for t in x)

            ax = fig.add_subplot(ROW, 1, i)
            _plot_row(ax, _x, y, c, i == 1, i == ROW)

        fig.suptitle(f"RTA in Japan Summer 2022 チャット頻出スタンプ・単語 ({npage}/{PAGES})",
                     color=FONT_COLOR, size="x-large")
        fig.text(0.03, 0.5, "単語 / 分 （同一メッセージ内の重複は除外）",
                 ha="center", va="center", rotation="vertical", color=FONT_COLOR, size="large")
        fig.savefig(f"{npage}.png", dpi=DPI)
        plt.close()
        print(npage)


def moving_average(x, w=AVR_WINDOW):
    _x = np.convolve(x, np.ones(w), "same") / w
    return _x[:len(x)]


def _plot_row(ax, x, y, total_raw, add_upper_legend, add_lower_legend):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=TIMEZONE))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(0, 60, 5)))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.set_facecolor(FACE_COLOR)
    for axis in ("top", "bottom", "left", "right"):
        ax.spines[axis].set_color(FRAME_COLOR)

    ax.tick_params(colors=FONT_COLOR, which="both")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, YMAX)

    total = moving_average(total_raw) * PER_SECONDS
    total = ax.fill_between(x, 0, total, color=BACKGROUND_COLOR)

    for i, game in enumerate(GAMES):
        if x[0] <= game.startat <= x[-1]:
            ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
            ax.annotate(game.name, xy=(game.startat, YMAX), xytext=(game.startat, YMAX * 0.85), verticalalignment="top",
                        color=FONT_COLOR, arrowprops=dict(facecolor=ARROW_COLOR, shrink=0.05), ha=game.align)

    # ys = []
    # labels = []
    # colors = []
    for words, style, color in RTA_EMOTES:
        if isinstance(words, str):
            words = (words, )
        _y = np.fromiter((sum(c[w] for w in words) for c in y), int)
        if not sum(_y):
            continue
        _y = moving_average(_y) * PER_SECONDS
        # ys.append(_y)
        # labels.append("\n".join(words))
        # colors.append(color if color else None)
        ax.plot(x, _y, label="\n".join(words), linestyle=style, color=(color if color else None))
    # ax.stackplot(x, ys, labels=labels, colors=colors)

    #
    avr_10min = moving_average(total_raw, FIND_WINDOW) * FIND_WINDOW
    words = Counter()
    for counter in y:
        words.update(counter)
    words = set(k for k, v in words.items() if v >= COUNT_THRESHOLD)
    words -= VOCABULARY

    annotations = []
    for word in words:
        at = []
        _ys = moving_average(np.fromiter((c[word] for c in y), int), FIND_WINDOW) * FIND_WINDOW
        for i, (_y, total_y) in enumerate(zip(_ys, avr_10min)):
            if _y >= total_y * DOMINATION_RATE and _y >= COUNT_THRESHOLD:
                at.append((i, _y * PER_SECONDS / FIND_WINDOW))
        if at:
            at.sort(key=lambda x: x[1])
            at = at[-1]

            if any(c in UNICODE_EMOJI for c in word):
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small", fontproperties=emoji_prop)
            else:
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small")
            annotations.append(text)
    adjust_text(annotations, only_move={"text": 'x'})

    if add_upper_legend:
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        _set_legend(leg)

    if add_lower_legend:
        leg = plt.legend([total], ["メッセージ / 分"], loc=(1.015, 0.4))
        _set_legend(leg)
        msg = "図中の単語は{}秒間で{}%の\nメッセージに含まれていた単語\n({:.1f}メッセージ / 秒 以上のもの)".format(
            FIND_WINDOW, int(DOMINATION_RATE * 100), COUNT_THRESHOLD / FIND_WINDOW
        )
        plt.gcf().text(0.915, 0.06, msg, fontsize="x-small", color=FONT_COLOR)


def _set_legend(leg):
    frame = leg.get_frame()
    frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(FRAME_COLOR)

    for text in leg.get_texts():
        text.set_color(FONT_COLOR)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+")
    args = parser.parse_args()

    timeline = _load_timeline(args.json)
    _save_counts(timeline)
    _plot(timeline)


if __name__ == "__main__":
    _main()
