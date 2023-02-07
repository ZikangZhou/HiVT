"""
Copied from evalai source files
Removed click library functions.

"""

from dataclasses import asdict, dataclass
from http import HTTPStatus
import json
import os
from os import PathLike
from os.path import expanduser
from pathlib import Path
from typing import BinaryIO, Optional

import requests
from tqdm import tqdm


class AmbigiousOption(Exception):
    pass


from enum import Enum


class URLS(Enum):
    login = "/api/auth/login"
    get_access_token = "/api/accounts/user/get_auth_token"
    challenge_list = "/api/challenges/challenge/all"
    past_challenge_list = "/api/challenges/challenge/past"
    future_challenge_list = "/api/challenges/challenge/future"
    challenge_details = "/api/challenges/challenge/{}"
    challenge_phase_details = "/api/challenges/challenge/phase/{}/"
    participant_teams = "/api/participants/participant_team"
    host_teams = "/api/hosts/challenge_host_team/"
    host_challenges = "/api/challenges/challenge_host_team/{}/challenge"
    challenge_phase_split_detail = "/api/challenges/{}/challenge_phase_split"
    create_host_team = "/api/hosts/create_challenge_host_team"
    host_team_list = "/api/hosts/challenge_host_team/"
    participant_challenges = "/api/participants/participant_team/{}/challenge"
    participant_team_list = "/api/participants/participant_team"
    participate_in_a_challenge = (
        "/api/challenges/challenge/{}/participant_team/{}"
    )
    challenge_phase_list = "/api/challenges/challenge/{}/challenge_phase"
    challenge_phase_detail = "/api/challenges/challenge/{}/challenge_phase/{}"
    my_submissions = "/api/jobs/challenge/{}/challenge_phase/{}/submission/"
    make_submission = "/api/jobs/challenge/{}/challenge_phase/{}/submission/"
    get_submission = "/api/jobs/submission/{}"
    leaderboard = "/api/jobs/challenge_phase_split/{}/leaderboard/"
    get_aws_credentials = (
        "/api/challenges/phases/{}/participant_team/aws/credentials/"
    )
    download_file = "/api/jobs/submission_files/?bucket={}&key={}"
    phase_details_using_slug = "/api/challenges/phase/{}/"
    get_presigned_url_for_annotation_file = (
        "/api/challenges/phases/{}/get_annotation_file_presigned_url/"
    )
    get_presigned_url_for_submission_file = (
        "/api/jobs/phases/{}/get_submission_file_presigned_url/"
    )
    finish_upload_for_submission_file = (
        "/api/jobs/phases/{}/finish_submission_file_upload/{}/"
    )
    finish_upload_for_annotation_file = (
        "/api/challenges/phases/{}/finish_annotation_file_upload/"
    )
    send_submission_message = "/api/jobs/phases/{}/send_submission_message/{}/"
    terms_and_conditions_page = "/web/challenges/challenge-page/{}/evaluation"


HOST_URL_FILE_NAME: str = "host_url"
AUTH_TOKEN_DIR: str = expanduser("~/.evalai/")
AUTH_TOKEN_FILE_NAME = "token.json"
HOST_URL_FILE_PATH: str = os.path.join(AUTH_TOKEN_DIR, HOST_URL_FILE_NAME)
API_HOST_URL = os.environ.get("EVALAI_API_URL", "https://eval.ai")
AUTH_TOKEN_PATH = os.path.join(AUTH_TOKEN_DIR, AUTH_TOKEN_FILE_NAME)
EVALAI_ERROR_CODES = (400, 401, 403, 406)


def get_host_url():
    """
    Returns the host url.
    """
    if not os.path.exists(HOST_URL_FILE_PATH):
        return API_HOST_URL
    else:
        with open(HOST_URL_FILE_PATH, "r") as fr:
            try:
                data = fr.read()
                return str(data)
            except (OSError, IOError) as e:
                raise OSError("Unable to read HOST_URL_FILE_PATH")


def get_user_auth_token():
    """
    Loads token to be used for sending requests.
    """
    if os.path.exists(AUTH_TOKEN_PATH):
        with open(str(AUTH_TOKEN_PATH), "r") as TokenObj:
            data = TokenObj.read()

        data = json.loads(data)
        token = data["token"]
        return token
    else:
        raise FileExistsError(
            "\nThe authentication token json file doesn't exists at the required path. "
            "Please download the file from the Profile section of the EvalAI webapp and "
            "place it at ~/.evalai/token.json\n",
        )


def get_request_header():
    """
    Returns user auth token formatted in header for sending requests.
    """
    header = {"Authorization": "Bearer {}".format(get_user_auth_token())}

    return header


def validate_token(response):
    """
    Function to check if the authentication token provided by user is valid or not.
    """
    if "detail" in response:
        if response["detail"] == "Invalid token":
            raise PermissionError(
                "\nThe authentication token you are using isn't valid."
                " Please generate it again.\n"
            )
        if response["detail"] == "Token has expired":
            raise PermissionError(response["detail"])


def get_submission_meta_attributes(challenge_id, phase_id):
    """
    Function to get all submission_meta_attributes for a challenge phase

    Parameters:
    challenge_id (int): id of challenge to which submission is made
    phase_id (int): id of challenge phase to which submission is made

    Returns:
    list: list of objects of submission meta attributes
    """
    url = "{}{}".format(get_host_url(), URLS.challenge_phase_detail.value)
    url = url.format(challenge_id, phase_id)
    headers = get_request_header()
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if response.status_code in EVALAI_ERROR_CODES:
            validate_token(response.json())
            raise PermissionError(
                "\nError: {}\n"
                "\nUse `evalai challenges` to fetch the active challenges.\n"
                "\nUse `evalai challenge CHALLENGE phases` to fetch the "
                "active phases.\n".format(response.json())
            )

        else:
            raise err
    except requests.exceptions.RequestException as err:
        raise err
    response = response.json()
    return response["submission_meta_attributes"]


def upload_file_to_s3(file, presigned_urls, max_chunk_size):
    """
    Function to upload a file, given the target presigned s3 url

    Arguments:
        file_name (str) -- the path of the file to be uploaded
        presigned_url (str) -- the presigned url to upload the file on s3"""
    parts = []
    index = 0
    file_size = Path(file.name).stat().st_size
    for chunk_size in tqdm(range(0, file_size, max_chunk_size)):
        presigned_url_object = presigned_urls[index]
        part = presigned_url_object["partNumber"]
        url = presigned_url_object["url"]
        file_data = file.read(max_chunk_size)
        response = requests.put(url, data=file_data)
        if response.status_code != HTTPStatus.OK:
            response.raise_for_status()

        etag = response.headers["ETag"]
        parts.append({"ETag": etag, "PartNumber": part})
        index += 1

    response = {"success": True, "parts": parts}
    return response


def publish_submission_message(challenge_phase_pk, submission_pk, headers):
    url = "{}{}".format(get_host_url(), URLS.send_submission_message.value)
    url = url.format(challenge_phase_pk, submission_pk)
    response = requests.post(
        url,
        headers=headers,
    )
    return response


def upload_file_using_presigned_url(
    challenge_phase_pk, file, file_type, submission_metadata={}
):
    if file_type == "submission":
        url = "{}{}".format(
            get_host_url(), URLS.get_presigned_url_for_submission_file.value
        )
        finish_upload_url = "{}{}".format(
            get_host_url(), URLS.finish_upload_for_submission_file.value
        )
    elif file_type == "annotation":
        url = "{}{}".format(
            get_host_url(), URLS.get_presigned_url_for_annotation_file.value
        )
        finish_upload_url = "{}{}".format(
            get_host_url(), URLS.finish_upload_for_annotation_file.value
        )
    url = url.format(challenge_phase_pk)
    headers = get_request_header()

    # Limit to max 100 MB chunk for multipart upload
    max_chunk_size = 20 * 1024 * 1024

    try:
        # Fetching the presigned url
        if file_type == "submission":
            file_size = Path(file.name).stat().st_size
            num_file_chunks = int(file_size / max_chunk_size) + 1
            data = {
                "status": "submitting",
                "file_name": file.name,
                "num_file_chunks": num_file_chunks,
            }
            data = dict(data, **asdict(submission_metadata))
            response = requests.post(url, headers=headers, data=data)

            if response.status_code is not HTTPStatus.CREATED:
                response.raise_for_status()

            # Update url params for multipart upload on S3
            finish_upload_url = finish_upload_url.format(
                challenge_phase_pk, response.json().get("submission_pk")
            )
        elif file_type == "annotation":
            file_size = Path(file.name).stat().st_size
            num_file_chunks = int(file_size / max_chunk_size) + 1

            data = {"file_name": file.name, "num_file_chunks": num_file_chunks}
            response = requests.post(url, headers=headers, data=data)
            if response.status_code is not HTTPStatus.OK:
                response.raise_for_status()

            # Update url params for multipart upload on S3
            finish_upload_url = finish_upload_url.format(challenge_phase_pk)

        response = response.json()
        presigned_urls = response.get("presigned_urls")
        upload_id = response.get("upload_id")
        if file_type == "submission":
            submission_pk = response.get("submission_pk")

        # Uploading the file to S3
        response = upload_file_to_s3(file, presigned_urls, max_chunk_size)

        if not response["success"] and file_type == "submission":
            # Publishing submission message to the message queue for processing
            response = publish_submission_message(
                challenge_phase_pk, submission_pk, headers
            )
            response.raise_for_status()

        data = {
            "parts": json.dumps(response.get("parts")),
            "upload_id": upload_id,
        }
        if file_type == "annotation":
            data["annotations_uploaded_using_cli"] = True

        # Complete multipart S3 upload
        upload_response = requests.post(
            finish_upload_url, headers=headers, data=data
        )

        if file_type == "submission":
            # Publishing submission message to the message queue for processing
            response = publish_submission_message(
                challenge_phase_pk, submission_pk, headers
            )
            response.raise_for_status()

        # Publish submission before throwing submission upload error
        if upload_response.status_code is not HTTPStatus.OK:
            upload_response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if response.status_code in EVALAI_ERROR_CODES:
            validate_token(response.json())
            if file_type == "submission":
                error_message = "\nThere was an error while making the submission: {}\n".format(
                    response.json()["error"]
                )
            elif file_type == "annotation":
                error_message = "\nThere was an error while uploading the annotation file: {}".format(
                    response.json()["error"]
                )
            raise requests.exceptions.HTTPError(error_message)
        else:
            raise err
    if file_type == "submission":
        success_message = "\nYour submission {} with the id {} is successfully submitted for evaluation.\n".format(
            file.name, submission_pk
        )
    elif file_type == "annotation":
        success_message = "\nThe annotation file {} for challenge phase {} is successfully uploaded.\n".format(
            file.name, challenge_phase_pk
        )


def make_submission(
    challenge_id,
    phase_id,
    file,
    submission_metadata={},
    submission_attribute_metadata={},
):
    """
    Function to submit a file to a challenge
    """
    url = "{}{}".format(get_host_url(), URLS.make_submission.value)
    url = url.format(challenge_id, phase_id)
    headers = get_request_header()
    input_file = {"input_file": file}
    data = {
        "status": "submitting",
        "submission_metadata": json.dumps(submission_attribute_metadata),
    }
    data = dict(data, **submission_metadata)
    try:
        response = requests.post(
            url, headers=headers, files=input_file, data=data
        )
        file.close()
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if response.status_code in EVALAI_ERROR_CODES:
            validate_token(response.json())
            raise requests.exceptions.HTTPError(
                "\nError: {}\n"
                "\nUse `evalai challenges` to fetch the active challenges.\n"
                "\nUse `evalai challenge CHALLENGE phases` to fetch the "
                "active phases.\n".format(response.json()["error"])
            )

        else:
            raise err
    response = response.json()

    return {
        "status": "success",
        "filename": file.name,
        "submission_id": response["id"],
    }


@dataclass
class SubmissionDetails:
    method_name: Optional[str]
    method_description: Optional[str]
    project_url: Optional[str]
    publication_url: Optional[str]


@dataclass
class SubmissionMetadata:
    is_public: Optional[str] = None
    method_name: Optional[str] = None
    method_description: Optional[str] = None
    project_url: Optional[str] = None
    publication_url: Optional[str] = None

    def __getitem__(self, key):
        return getattr(self, key)


def submit(
    phase_id: int,
    challenge_id: int,
    file: BinaryIO,
    annotation: bool,
    large: bool,
    public: bool,
    private: bool,
    submission_details: SubmissionDetails = None,
):
    """
    For uploading submission files to evalai:
        - Invoked by running 'evalai challenge CHALLENGE phase PHASE submit --file FILE'
        - For large files, add a '--large' option at the end of the command

    For uploading test annotation files to evalai:
        - Invoked by running "evalai challenge CHALLENGE phase PHASE submit --file FILE --annotation"

    Arguments:
        file (str) -- the path of the file to be uploaded
        annotations (boolean) -- flag to denote if file is a test annotation file
        large (boolean) -- flag to denote if submission file is large (if large, presigned urls are used for uploads)
        public (boolean) -- flag to denote if submission is public
        private (boolean) -- flag to denote if submission is private
    Returns:
        None
    """
    if public and private:
        message: str = "\nError: Submission can't be public and private.\nPlease select either --public or --private"
        raise AmbigiousOption(message)
    else:
        if annotation:
            upload_file_using_presigned_url(phase_id, file, "annotation")
        else:
            submission_metadata: SubmissionMetadata = SubmissionMetadata()
            if public:
                submission_metadata.is_public = json.dumps(True)
            elif private:
                submission_metadata.is_public = json.dumps(False)
            else:
                submission_metadata.is_public = None
            if submission_details is not None:
                submission_metadata.method_name = (
                    submission_details.method_name
                )
                submission_metadata.method_description = (
                    submission_details.method_description
                )
                submission_metadata.project_url = (
                    submission_details.project_url
                )
                submission_metadata.publication_url = (
                    submission_details.publication_url
                )

            if large:
                upload_file_using_presigned_url(
                    phase_id, file, "submission", submission_metadata
                )
            else:
                make_submission(
                    challenge_id,
                    phase_id,
                    file,
                    submission_metadata,
                    None,
                )
