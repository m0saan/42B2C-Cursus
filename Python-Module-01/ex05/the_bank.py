class Account:
    ID_COUNT = 1

    def __init__(self, name, **kwargs):
        self.__dict__.update(kwargs)
        self.id = self.ID_COUNT
        Account.ID_COUNT += 1
        self.name = name
        if not hasattr(self, 'value'):
            self.value = 0
        if self.value < 0:
            raise AttributeError("Attribute value cannot be negative.")
        if not isinstance(self.name, str):
            raise AttributeError("Attribute name must be a str object.")

    def transfer(self, amount):
        self.value += amount

class Bank:
    def __init__(self):
        self.accounts = []

    def add(self, new_account):
        """ Add new_account in the Bank
            @new_account:  Account() new account to append
            @return   True if success, False if an error occurred
        """
        if not isinstance(new_account, Account):
            return False
        if not hasattr(new_account, 'name'):
            return False
        for account in self.accounts:
            if account.name == new_account.name:
                return False
        if not self.verify_account(new_account):
            return False
        self.accounts.append(new_account)
        return True

    def transfer(self, origin, dest, amount):
        """ Perform the fund transfer
            @origin:  str(name) of the first account
            @dest:    str(name) of the destination account
            @amount:  float(amount) amount to transfer
            @return   True if success, False if an error occurred
        """
        origin_account = self.find_account(origin)
        dest_account = self.find_account(dest)
        if origin_account is None or dest_account is None:
            return False
        if origin == dest:
            return True
        if amount < 0 or amount > origin_account.value:
            return False
        origin_account.transfer(-amount)
        dest_account.transfer(amount)
        return True

    def fix_account(self, name):
        """ Fix account associated to name if corrupted
            @name:   str(name) of the account
            @return  True if success, False if an error occurred
        """
        if not isinstance(name, str):
            return False
        for account in self.accounts:
            if account.name == name:
                if self.verify_account(account):
                    return True
                else:
                    account.__class__ = Account
                    return True
        return False

    def verify_account(self, account):
        """ Verify if an account is valid
            @account:  Account object
            @return:   True if account is valid, False otherwise
        """
        if len(account.__dict__) % 2 == 1:
            return False
        if any(attr.startswith('b') for attr in account.__dict__):
            return False
        if any(attr.startswith('zip') or attr.startswith('addr') for attr in account.__dict__):
            return False
        if not hasattr(account, 'name') or not hasattr(account, 'id') or not hasattr(account, 'value'):
            return False
        if not isinstance(account.name, str) or not isinstance(account.id, int) \
                or not isinstance(account.value, (int, float)):
            return False
        return True

    def find_account(self, name):
        """ Find account with given name
            @name:   str(name) of the account
            @return: Account object if found, None otherwise
        """
        for account in self.accounts:
            if account.name == name:
                return account
        return None
