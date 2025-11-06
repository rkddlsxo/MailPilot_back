# 단순화된 설정 구조 정의
SETTINGS_STRUCTURE = {
    'GENERAL': {
        'READ': {
            'name': '읽기',
            'sections': {
                'EMAIL_FETCH_SETTINGS': {
                    'name': '메일 가져오기 설정',
                    'fields': {
                        'gmailFetchCount': {
                            'label': 'Gmail에서 가져오는 메일 수 (3~100개)',
                            'type': 'number',
                            'default': 5,
                            'min': 3,
                            'max': 100
                        },
                        'itemsPerPage': {
                            'label': '페이지당 표시할 메일 수 (3~50개)',
                            'type': 'number',
                            'default': 10,
                            'min': 3,
                            'max': 50
                        }
                    }
                }
            }
        },
        'WRITE': {
            'name': '쓰기',
            'sections': {
                'DEFAULT_FONT': {
                    'name': '기본 폰트',
                    'fields': {
                        'fontFamily': {
                            'label': '글꼴',
                            'type': 'select',
                            'options': [
                                {'value': 'system', 'label': '시스템 기본'},
                                {'value': 'Arial', 'label': 'Arial'},
                                {'value': '돋움', 'label': '돋움'},
                                {'value': '맑은 고딕', 'label': '맑은 고딕'},
                                {'value': '굴림', 'label': '굴림'},
                                {'value': '바탕', 'label': '바탕'},
                                {'value': '궁서', 'label': '궁서'},
                                {'value': 'Times New Roman', 'label': 'Times New Roman'},
                                {'value': 'Helvetica', 'label': 'Helvetica'},
                                {'value': 'Verdana', 'label': 'Verdana'},
                                {'value': 'Georgia', 'label': 'Georgia'},
                                {'value': 'Courier New', 'label': 'Courier New'}
                            ],
                            'default': 'system'
                        },
                        'fontSize': {
                            'label': '크기',
                            'type': 'select',
                            'options': [
                                {'value': '10px', 'label': '10px (매우 작게)'},
                                {'value': '11px', 'label': '11px (작게)'},
                                {'value': '12px', 'label': '12px (작게)'},
                                {'value': '13px', 'label': '13px (보통)'},
                                {'value': '14px', 'label': '14px (보통)'},
                                {'value': '15px', 'label': '15px (크게)'},
                                {'value': '16px', 'label': '16px (크게)'},
                                {'value': '18px', 'label': '18px (더 크게)'},
                                {'value': '20px', 'label': '20px (매우 크게)'},
                                {'value': '22px', 'label': '22px (최대)'}
                            ],
                            'default': '14px'
                        }
                    }
                },
                'SENDER_INFO': {
                    'name': '보내는 이름',
                    'fields': {
                        'senderName': {
                            'label': '보내는 이름 (선택사항)',
                            'type': 'text',
                            'default': '',
                            'placeholder': '최수운'
                        }
                    }
                }
            }
        },
        'THEME': {
            'name': '테마',
            'sections': {
                'APPEARANCE': {
                    'name': '화면 테마',
                    'fields': {
                        'appearance': {
                            'label': '테마 모드',
                            'type': 'radio',
                            'options': [
                                {'value': 'light', 'label': '☀️ 라이트 모드'},
                                {'value': 'dark', 'label': '🌙 다크 모드'},
                                {'value': 'auto', 'label': '🔄 시스템 설정 따르기'}
                            ],
                            'default': 'light'
                        }
                    }
                }
            }
        }
    },
    'MY_EMAIL': {
        'SIGNATURE_MANAGEMENT': {
            'name': '서명 관리',
            'sections': {
                'SIGNATURE_ADD': {
                    'name': '서명 관리',
                    'fields': {
                        'signatures': {
                            'label': '서명 추가',
                            'type': 'signature_list',
                            'default': []
                        }
                    }
                }
            }
        }
    }
}

def get_default_settings():
    """모든 기본 설정값 추출"""
    defaults = {}
    
    for category, category_data in SETTINGS_STRUCTURE.items():
        defaults[category] = {}
        for subcategory, subcategory_data in category_data.items():
            defaults[category][subcategory] = {}
            if 'sections' in subcategory_data:
                for section_key, section_data in subcategory_data['sections'].items():
                    for field_key, field_data in section_data['fields'].items():
                        defaults[category][subcategory][field_key] = field_data.get('default')
    
    return defaults

def get_field_info(category, subcategory, field_name):
    """특정 필드의 정보 가져오기"""
    if category in SETTINGS_STRUCTURE:
        if subcategory in SETTINGS_STRUCTURE[category]:
            subcategory_data = SETTINGS_STRUCTURE[category][subcategory]
            if 'sections' in subcategory_data:
                for section_data in subcategory_data['sections'].values():
                    if field_name in section_data['fields']:
                        return section_data['fields'][field_name]
    return None