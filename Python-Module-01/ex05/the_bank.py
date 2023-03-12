# in the_bank.py

class Account(object):
    ID_COUNT = 1

    def __init__(self, name, **kwargs):
        self.id = self.ID_COUNT
        self.name = name
        self.__dict__.update(kwargs)
        # if hasattr(self, 'value'):
        #     self.value = 0
        Account.ID_COUNT += 1

    def transfer(self, amount):
        self.value += amount

    def is_corrupted(self):
        ok = True
        attributes = list(self.__dict__.keys())
        if "value" not in attributes:
            return True
        if len(attributes) % 2 == 0:
            return True
        for attr in attributes:
            if attr[0] == "b":
                return True
            if attr.startswith("zip") or attr.startswith("addr"):
                ok = False
        return ok


# in the_bank.py

class Bank(object):
    """The bank"""

    def __init__(self):
        self.accounts = []
        self.has_b_prefix = False

    def add(self, new_account):
        if not isinstance(new_account, Account):
            return False
        if not hasattr(new_account, 'name'):
            return False
        for account in self.accounts:
            if account.name == new_account.name:
                return False
        self.accounts.append(new_account)

    def get_account(self, p):
        if isinstance(p, int):
            for account in self.accounts:
                if account.id == p:
                    return account
        if isinstance(p, str):
            for account in self.accounts:
                if account.name == p:
                    return account

    def transfer(self, origin, dest, amount: float) -> bool:
        """
        @origin:  int(id) or str(name) of the first account
        @dest:    int(id) or str(name) of the destination account
        @amount:  float(amount) amount to transfer
        @return         True if success, False if an error occurred
        """
        origin_account = self.get_account(origin)
        dest_account = self.get_account(dest)
        if not isinstance(origin_account, Account) or not isinstance(dest_account, Account):
            return False
        elif not origin_account.is_corrupted() and not dest_account.is_corrupted():
            value = origin_account.value
            if value >= amount and value > 0:
                origin_account.value -= amount
                dest_account.transfer(amount)
                return True
        return False

    def fix_account(self, p) -> bool:
        """
            fix the corrupted account
            @account: int(id) or str(name) of the account
            @return         True if success, False if an error occurred
        """

        account = self.get_account(p)

        if isinstance(account, Account):
            if account.is_corrupted():
                attributes = list(account.__dict__.keys())
                if 'name' not in attributes:
                    account.__dict__.update({'name': p})
                if 'value' not in attributes:
                    account.__dict__.update({'value': 0})
                if 'addr' not in attributes:
                    account.__dict__.update({'zip': ''})
                if 'addr' not in attributes:
                    account.__dict__.update({'addr': ''})

                for attr in attributes:
                    if attr.startswith('b'):
                        tmp = attr
                        while tmp.startswith('b'):
                            tmp = tmp[1:]
                        account.__dict__[tmp] = account.__dict__[attr]
                        del account.__dict__[attr]

                if account.__dict__.__len__() % 2 == 0:
                    return False

            if account.is_corrupted():
                return False
            return True