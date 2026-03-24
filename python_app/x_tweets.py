import os
import requests
from dotenv import load_dotenv

load_dotenv()


class TweetCollector:
    """Class to collect tweets from X (formerly Twitter) API."""

    _bearer_token: str
    _headers: dict
    _params: dict
    _base_url: str

    def __init__(
        self,
        x_user_id: str,
        from_date_time: str,
        to_date_time: str,
        next_token: str = "",
    ):
        """
        Creates an instance of TweetCollector to collect tweets from X API for a specific user and time range.

        :param self: The instance of the class.
        :type self: TweetCollector
        :param x_user_id: The user ID of the X account from which to collect tweets. For example, for Elon Musk's account (@elonmusk), the user ID is 59773459.
        :type x_user_id: str
        :param from_date_time: The oldest UTC timestamp from which the Tweets will be provided. YYYY-MM-DDTHH:mm:ssZ (ISO 8601/RFC 3339).
        :type from_date_time: str
        :param to_date_time: The newest, most recent UTC timestamp to which the Tweets will be provided. YYYY-MM-DDTHH:mm:ssZ (ISO 8601/RFC 3339).
        :type to_date_time: str
        :param next_token: The token for fetching the next page of results.
        :type next_token: str
        """

        # Initialize the TweetCollector with the provided bearer token
        self._bearer_token = os.getenv("TWITTER_ACCESS_TOKEN", "")

        # Set the base URL for the X API endpoint to fetch tweets from a specific user (user ID: 59773459)
        self._base_url = f"https://api.x.com/2/users/{x_user_id}/tweets"

        # Headers
        self._headers = {"Authorization": f"Bearer {self._bearer_token}"}

        self._params = {
            "tweet.fields": "created_at,note_tweet,author_id,public_metrics,lang,source,entities,context_annotations,geo",
            "max_results": 100,
            "user.fields": "name,username,location,description,public_metrics",
            "start_time": from_date_time,
            "end_time": to_date_time,
        }

        if next_token:
            self._params["pagination_token"] = next_token

    def __get_url__(self) -> str:
        """Constructs the URL for the X API request based on the user ID and parameters."""
        query_params = "&".join(
            [f"{key}={value}" for key, value in self._params.items()]
        )
        self._base_url = f"{self._base_url}?{query_params}"
        return self._base_url

    def make_request(self) -> dict:
        """Makes a request to the X API with the given parameters and returns the response as a dictionary."""

        response = requests.get(
            url=self._base_url,
            headers=self._headers,
            params=self._params,
        )
        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )
        return response.json()


# if __name__ == "__main__":
#     # Example usage
#     x_user_id = "59773459"  # @infomoney
#     from_date_time = "2016-01-01T03:00:00Z"
#     to_date_time = "2025-12-31T02:59:00Z"

#     collector = TweetCollector(x_user_id, from_date_time, to_date_time)
#     tweets_data = collector.make_request()
#     print(tweets_data)
