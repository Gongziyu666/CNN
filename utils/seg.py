import pkuseg
import jieba
# jieba.load_userdict('dict.txt')


class PkusegUtils:

    def __init__(self):
        self.pk = pkuseg.pkuseg()

    @staticmethod
    def remove_syb(words):
        unused_words = u" \t\r\n，。：；“‘”【】『』|=+-——（）*&……%￥#@！~·《》？/?<>,.;:'\"[]{}_)(^$!`"
        unused_english = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        for i in unused_words:
            words = words.replace(i, "")
        for i in unused_english:
            words = words.replace(i, "")
        return words

    def seg(self, words):
        words = self.remove_syb(words)
        return self.pk.cut(words)


class JiebaUtils:

    @staticmethod
    def remove_syb(words):
        unused_words = u" \t\r\n，。：；“‘”【】『』|=+-——（）*&……%￥#@！~·《》？/?<>,.;:'\"[]{}_)(^$!`"
        unused_english = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        for i in unused_words:
            words = words.replace(i, "")
        for i in unused_english:
            words = words.replace(i, "")
        return words

    def seg(self, words):
        words = self.remove_syb(words)
        return [x for x in jieba.cut(words)]
