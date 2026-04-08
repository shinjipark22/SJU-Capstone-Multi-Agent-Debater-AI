"""
source_config.py — 출처 메타데이터 설정

각 출처(source)의 도메인 커버리지, 공신력 등급, 편향 유형을 정의한다.
retrieval 시 가중치 기반 점수 산정에 사용된다.

[credibility_tier]
    1: 국제기구, 피어리뷰 학술지 — 최고 공신력
    2: 싱크탱크, 연구기관 — 높은 공신력
    3: 주요 언론사 — 보통 공신력

[bias_type]
    institutional: 국제기구 (mandate 기반 관점)
    academic:      학술 연구 (데이터/방법론 기반)
    think_tank:    싱크탱크 (정책 분석 기반)
    advocacy:      인권/환경 NGO (옹호 기반)
    media:         언론사 (보도 기반)

[도메인 가중치 — retrieval 시 적용]
    primary_domain 일치:     1.0
    secondary_domains 일치:  0.7
    기타 (cross-domain):     0.4
"""

from __future__ import annotations

from typing import Dict, List


# ── 출처별 메타데이터 ─────────────────────────────────────────────────────────
# key: domain 문자열 (source_url에서 매칭)
# value: {primary_domain, secondary_domains, credibility_tier, bias_type}

SOURCE_METADATA: Dict[str, Dict] = {
    # ── 국제기구 (tier 1) ─────────────────────────────────
    "weforum.org": {
        "primary_domain": "tech",
        "secondary_domains": ["econ", "poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "oecd.org": {
        "primary_domain": "econ",
        "secondary_domains": ["tech", "poli", "env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "imf.org": {
        "primary_domain": "econ",
        "secondary_domains": ["poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "worldbank.org": {
        "primary_domain": "econ",
        "secondary_domains": ["poli", "env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "ilo.org": {
        "primary_domain": "econ",
        "secondary_domains": ["tech", "poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "who.int": {
        "primary_domain": "poli",
        "secondary_domains": ["env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "fao.org": {
        "primary_domain": "env",
        "secondary_domains": ["econ", "poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "wfp.org": {
        "primary_domain": "econ",
        "secondary_domains": ["poli", "env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "unep.org": {
        "primary_domain": "env",
        "secondary_domains": ["poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "undp.org": {
        "primary_domain": "poli",
        "secondary_domains": ["econ", "env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "unhcr.org": {
        "primary_domain": "poli",
        "secondary_domains": ["env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "iea.org": {
        "primary_domain": "env",
        "secondary_domains": ["tech", "econ"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "ipcc.ch": {
        "primary_domain": "env",
        "secondary_domains": ["poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "wto.org": {
        "primary_domain": "econ",
        "secondary_domains": ["poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "internal-displacement.org": {
        "primary_domain": "poli",
        "secondary_domains": ["env"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "ecb.europa.eu": {
        "primary_domain": "econ",
        "secondary_domains": ["poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },
    "energy.ec.europa.eu": {
        "primary_domain": "env",
        "secondary_domains": ["tech", "poli"],
        "credibility_tier": 1,
        "bias_type": "institutional",
    },

    # ── 학술지 (tier 1) ──────────────────────────────────
    "nature.com": {
        "primary_domain": "tech",
        "secondary_domains": ["env"],
        "credibility_tier": 1,
        "bias_type": "academic",
    },
    "sciencedirect.com": {
        "primary_domain": "tech",
        "secondary_domains": ["econ", "env"],
        "credibility_tier": 1,
        "bias_type": "academic",
    },
    "thelancet.com": {
        "primary_domain": "env",
        "secondary_domains": ["poli"],
        "credibility_tier": 1,
        "bias_type": "academic",
    },
    "pmc.ncbi.nlm.nih.gov": {
        "primary_domain": "tech",
        "secondary_domains": ["env"],
        "credibility_tier": 1,
        "bias_type": "academic",
    },
    "arxiv.org": {
        "primary_domain": "tech",
        "secondary_domains": [],
        "credibility_tier": 1,
        "bias_type": "academic",
    },
    "springer.com": {
        "primary_domain": "tech",
        "secondary_domains": ["env"],
        "credibility_tier": 1,
        "bias_type": "academic",
    },

    # ── 싱크탱크 (tier 2) ────────────────────────────────
    "brookings.edu": {
        "primary_domain": "poli",
        "secondary_domains": ["econ", "tech"],
        "credibility_tier": 2,
        "bias_type": "think_tank",
    },
    "sipri.org": {
        "primary_domain": "poli",
        "secondary_domains": ["econ"],
        "credibility_tier": 2,
        "bias_type": "think_tank",
    },
    "csis.org": {
        "primary_domain": "poli",
        "secondary_domains": ["econ"],
        "credibility_tier": 2,
        "bias_type": "think_tank",
    },
    "piie.com": {
        "primary_domain": "econ",
        "secondary_domains": ["poli"],
        "credibility_tier": 2,
        "bias_type": "think_tank",
    },
    "mckinsey.com": {
        "primary_domain": "tech",
        "secondary_domains": ["econ"],
        "credibility_tier": 2,
        "bias_type": "think_tank",
    },

    # ── 인권/옹호 기관 (tier 2) ──────────────────────────
    "amnesty.org": {
        "primary_domain": "poli",
        "secondary_domains": [],
        "credibility_tier": 2,
        "bias_type": "advocacy",
    },
    "hrw.org": {
        "primary_domain": "poli",
        "secondary_domains": [],
        "credibility_tier": 2,
        "bias_type": "advocacy",
    },
    "rsf.org": {
        "primary_domain": "poli",
        "secondary_domains": [],
        "credibility_tier": 2,
        "bias_type": "advocacy",
    },
    "aclu.org": {
        "primary_domain": "poli",
        "secondary_domains": [],
        "credibility_tier": 2,
        "bias_type": "advocacy",
    },

    # ── 한국 공공기관 (tier 2) ────────────────────────────
    "bok.or.kr": {
        "primary_domain": "econ",
        "secondary_domains": [],
        "credibility_tier": 2,
        "bias_type": "institutional",
    },
    "kdi.re.kr": {
        "primary_domain": "econ",
        "secondary_domains": ["poli"],
        "credibility_tier": 2,
        "bias_type": "think_tank",
    },
    "kostat.go.kr": {
        "primary_domain": "econ",
        "secondary_domains": ["poli"],
        "credibility_tier": 2,
        "bias_type": "institutional",
    },
}

# ── 기본값 (매핑에 없는 출처) ─────────────────────────────────────────────────
_DEFAULT_METADATA: Dict = {
    "primary_domain": "",
    "secondary_domains": [],
    "credibility_tier": 3,
    "bias_type": "media",
}


def get_source_metadata(url: str) -> Dict:
    """URL에서 출처 메타데이터를 조회한다.

    Args:
        url: 기사/보고서 URL

    Returns:
        {primary_domain, secondary_domains, credibility_tier, bias_type}
    """
    for domain, meta in SOURCE_METADATA.items():
        if domain in url:
            return {
                "primary_domain": meta["primary_domain"],
                "secondary_domains": ",".join(meta["secondary_domains"]),
                "credibility_tier": meta["credibility_tier"],
                "bias_type": meta["bias_type"],
            }
    return {
        "primary_domain": _DEFAULT_METADATA["primary_domain"],
        "secondary_domains": "",
        "credibility_tier": _DEFAULT_METADATA["credibility_tier"],
        "bias_type": _DEFAULT_METADATA["bias_type"],
    }


# ── Retrieval 가중치 설정 ────────────────────────────────────────────────────

DOMAIN_WEIGHTS = {
    "primary": 1.0,
    "secondary": 0.7,
    "cross": 0.4,
}

CREDIBILITY_WEIGHTS = {
    1: 1.0,
    2: 0.85,
    3: 0.7,
}

# Multi-domain 슬롯 비율
MULTI_DOMAIN_RATIO = {
    "primary": 0.6,
    "cross": 0.3,
    "other": 0.1,
}
