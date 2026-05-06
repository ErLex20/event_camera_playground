"""
Event Detector utils implementation.

dotX Automation s.r.l. <info@dotxautomation.com>

May 6, 2026
"""

# Copyright 2025 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Lock

class AtomicBool:
    """
    Class to represent a boolean value that can be safely accessed and modified.
    """

    def __init__(self, initial: bool = False) -> None:
        """
        Constructor.

        :param initial: Initial value of the boolean, defaults to False.
        """
        self._value = initial
        self._lock = Lock()

    def store(self, new_value: bool) -> None:
        """
        Sets the boolean value in a thread-safe manner.

        :param new_value: The new boolean value to be stored.
        """
        with self._lock:
            self._value = new_value

    def load(self) -> bool:
        """
        Retrieves the current boolean value in a thread-safe manner.

        :return: The current boolean value.
        """
        with self._lock:
            return self._value
