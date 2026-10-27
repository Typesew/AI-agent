"""Правила неоднозначности при распознавании сущностей."""

import unittest

from rag.entities import EntityRecognizer


class EntitiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = EntityRecognizer()

    def products(self, q):
        return set(self.r.recognize(q).products)

    def test_longest_match_wins(self):
        self.assertEqual(self.products("Студенческая Северная карта: сколько стоит?"), {"sever_student"})
        self.assertEqual(self.products("Что по Северной Карте?"), {"sever_debit"})

    def test_product_beats_bank(self):
        e = self.r.recognize("Сравни «Опорная 55» и «Меридиан 150»")
        self.assertEqual(set(e.products), {"opora_credit", "merid_credit"})
        self.assertEqual(e.banks, {})

    def test_bank_and_type_resolve_to_product(self):
        self.assertEqual(self.products("Расскажи про рассрочку в банке Кедр"), {"kedr_install"})
        self.assertEqual(self.products("Какая кредитка у Меридиана?"), {"merid_credit"})

    def test_bank_only_means_all_products(self):
        self.assertEqual(self.products("Что есть у банка Кедр?"), {"kedr_credit", "kedr_install", "kedr_debit", "kedr_student"})

    def test_number_phrase_needs_context(self):
        self.assertEqual(self.products("Успею вернуть долг за 100 дней?"), set())
        self.assertEqual(self.products("Кредитка «100 дней»: какая комиссия?"), {"balt_credit"})

    def test_generic_phrase_is_not_product(self):
        e = self.r.recognize("Можно ли платить частями?")
        self.assertEqual(e.products, {})
        self.assertIn("term_installment", e.type_terms)
        self.assertEqual(self.products("Что будет, если пропущу платёж по «Плати частями»?"), {"sever_install"})

    def test_product_name_is_not_term(self):
        e = self.r.recognize("Чем «Кедр Рассрочка» отличается от обычной рассрочки?")
        self.assertEqual(e.type_terms, {})
        self.assertEqual(e.content_terms, {})

    def test_product_words_do_not_eat_events(self):
        e = self.r.recognize("Что будет, если пропущу платёж по «Плати частями», и что при двух пропусках?")
        self.assertEqual(set(e.events), {"ev_missed_payment", "ev_missed_two"})

    def test_terms_and_events(self):
        e = self.r.recognize("Какие кредитные карты сохраняют льготный период при снятии наличных?")
        self.assertIn("term_credit_card", e.type_terms)
        self.assertIn("term_grace", e.content_terms)
        self.assertIn("ev_cash", e.events)

    def test_domain_flag(self):
        self.assertFalse(self.r.recognize("Какая погода завтра?").domain)
        calc = self.r.recognize("Кредит 100 000 ₽ на 2 года под 18%, какая переплата?")
        self.assertTrue(calc.domain)
        self.assertTrue(calc.numbers)


if __name__ == "__main__":
    unittest.main()
