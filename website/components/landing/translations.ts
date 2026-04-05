export type Lang = 'ru' | 'en';

export const translations = {
  ru: {
    heroTitle: 'engram',
    heroSubtitle: 'AI-память для агентов',
    heroButton: 'Начать',

    featuresTitle: 'Возможности',
    featureHybridSearch: 'Гибридный поиск',
    featureHybridSearchDescription:
      'Комбинация векторного поиска и FTS5 для максимальной точности. Семантика и ключевые слова работают вместе.',
    featureSelfLearning: 'Самообучение',
    featureSelfLearningDescription:
      'Q-Learning роутер автоматически выбирает лучшую стратегию поиска на основе предыдущих результатов.',
    featureConsolidation: 'Консолидация',
    featureConsolidationDescription:
      'Дедупликация и слияние памяти. Система сама находит похожие записи и объединяет их.',
    featureQualityScoring: 'Оценка качества',
    featureQualityScoringDescription:
      'Эвристический анализ и LLM-судья оценивают релевантность каждого результата поиска.',
    featureInsights: 'Инсайты',
    featureInsightsDescription:
      'Кластеризация и временные паттерны. Система выявляет связи, которые не видны при простом поиске.',
    featureCrossProject: 'Кросс-проект',
    featureCrossProjectDescription:
      'Перенос знаний между проектами. Опыт из одного проекта доступен во всех остальных.',

    howItWorksTitle: 'Как это работает',
    stepStore: 'Сохранение',
    stepStoreDescription: 'Агент сохраняет контекст, решения и факты в память',
    stepSearch: 'Поиск',
    stepSearchDescription: 'Гибридный поиск находит релевантные воспоминания',
    stepJudge: 'Оценка',
    stepJudgeDescription: 'LLM-судья оценивает качество и релевантность',
    stepLearn: 'Обучение',
    stepLearnDescription: 'Q-Learning оптимизирует стратегии поиска',

    quickStartTitle: 'Быстрый старт',

    ctaTitle: 'Готовы начать?',
    ctaSubtitle: 'Дайте вашим AI-агентам долгосрочную память',
    ctaDocs: 'Документация',
    ctaGithub: 'GitHub',

    footerCopyright: 'engram © 2026',
    footerDocs: 'Документация',
    footerGithub: 'GitHub',
  },
  en: {
    heroTitle: 'engram',
    heroSubtitle: 'AI memory for agents',
    heroButton: 'Get Started',

    featuresTitle: 'Features',
    featureHybridSearch: 'Hybrid Search',
    featureHybridSearchDescription:
      'Vector search combined with FTS5 for maximum accuracy. Semantics and keywords work together.',
    featureSelfLearning: 'Self-Learning',
    featureSelfLearningDescription:
      'Q-Learning router automatically selects the best search strategy based on previous results.',
    featureConsolidation: 'Consolidation',
    featureConsolidationDescription:
      'Deduplication and memory merging. The system finds similar entries and combines them automatically.',
    featureQualityScoring: 'Quality Scoring',
    featureQualityScoringDescription:
      'Heuristic analysis and LLM judge evaluate the relevance of every search result.',
    featureInsights: 'Insights',
    featureInsightsDescription:
      'Clustering and temporal patterns. The system discovers connections invisible to simple search.',
    featureCrossProject: 'Cross-Project',
    featureCrossProjectDescription:
      'Knowledge transfer between projects. Experience from one project is available across all others.',

    howItWorksTitle: 'How It Works',
    stepStore: 'Store',
    stepStoreDescription: 'Agent saves context, decisions, and facts to memory',
    stepSearch: 'Search',
    stepSearchDescription: 'Hybrid search finds relevant memories',
    stepJudge: 'Judge',
    stepJudgeDescription: 'LLM judge evaluates quality and relevance',
    stepLearn: 'Learn',
    stepLearnDescription: 'Q-Learning optimizes search strategies',

    quickStartTitle: 'Quick Start',

    ctaTitle: 'Ready to start?',
    ctaSubtitle: 'Give your AI agents long-term memory',
    ctaDocs: 'Documentation',
    ctaGithub: 'GitHub',

    footerCopyright: 'engram © 2026',
    footerDocs: 'Documentation',
    footerGithub: 'GitHub',
  },
} as const;

export function useTranslations(lang: string) {
  const locale = lang === 'ru' ? 'ru' : 'en';
  return translations[locale];
}
